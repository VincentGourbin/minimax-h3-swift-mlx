// Qwen3VLTextEncoder.swift - Qwen3-VL-32B text stack as MiniMax-H3's conditioner
// Copyright 2026 Vincent Gourbin
//
// MiniMax-H3 conditions on the *unnormalized* hidden state after decoder layer 50 of its Qwen3-VL
// conditioner (hidden_states[50], with hidden_states[0] the embedding output). The final norm and
// LM head are never used, so we materialize layers 0..<50 plus the embedding — the last 14
// layers, the norm and the head are dead weight.
//
// Interleaved mrope: each of the 64 frequency channels belongs to one of (t, h, w) — t owns
// indices {0,3,...,57} ∪ {60..63} (24 channels), h owns {1,4,...,58}, w owns {2,5,...,59}
// (20 each; sections [24, 20, 20]). Text runs advance all three axes together (which collapses
// to standard 1-D RoPE at theta 5e6 for pure text); an image block freezes t and lays h/w out
// over its merged token grid, and the running position then advances by max(gridH, gridW)/merge
// — NOT by the token count. For fl2va, vision-tower embeds replace the `<|image_pad|>` token
// embeddings and the three deepstack taps are added after decoder layers 0/1/2 at those same
// positions (DeepStack).

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Qwen3VLTextConfig: Codable, Sendable {
    public var hiddenSize = 5120
    public var numHiddenLayers = 64
    public var numAttentionHeads = 64
    public var numKeyValueHeads = 8
    public var headDim = 128
    public var intermediateSize = 25600
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 5_000_000
    public var vocabSize = 151_936

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
    }

    public init() {}

    /// Reads the nested `text_config` from the repo's `text_encoder/config.json`.
    public static func load(from url: URL) throws -> Qwen3VLTextConfig {
        struct Wrapper: Codable { let text_config: Qwen3VLTextConfig }
        return try JSONDecoder().decode(Wrapper.self, from: Data(contentsOf: url)).text_config
    }
}

// MARK: - Decoder layer

final class Qwen3VLAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: Qwen3VLTextConfig) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    /// x (1, L, hidden); rope cos/sin (L, headDim/2) applied rotate_half style over full headDim.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, seq) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(batch, seq, heads, headDim))
        var k = kNorm(kProj(x).reshaped(batch, seq, kvHeads, headDim))
        let v = vProj(x).reshaped(batch, seq, kvHeads, headDim).transposed(0, 2, 1, 3)

        q = applyNeoxRope(q, cos: cos, sin: sin).transposed(0, 2, 1, 3)
        k = applyNeoxRope(k, cos: cos, sin: sin).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: .causal
        )
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, seq, heads * headDim))
    }

    /// Standard GPT-NeoX rotate-half RoPE over the full head dim; x (B, L, H, D).
    private func applyNeoxRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = headDim / 2
        let cosB = cos.asType(x.dtype).reshaped(1, cos.dim(0), 1, half)
        let sinB = sin.asType(x.dtype).reshaped(1, sin.dim(0), 1, half)
        let x1 = x[.ellipsis, ..<half]
        let x2 = x[.ellipsis, half...]
        return concatenated(
            [x1 * cosB - x2 * sinB, x2 * cosB + x1 * sinB],
            axis: -1
        )
    }
}

final class Qwen3VLMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3VLTextConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class Qwen3VLDecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: Qwen3VLAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen3VLMLP

    init(config: Qwen3VLTextConfig) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attention.wrappedValue = Qwen3VLAttention(config: config)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = Qwen3VLMLP(config: config)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var x = x + attention(inputNorm(x), cos: cos, sin: sin)
        x = x + mlp(postAttentionNorm(x))
        return x
    }
}

// MARK: - Multimodal layout

/// Everything the text stack needs to know about the vision blocks of an fl2va presentation:
/// the 3-axis rotary positions of every token and where the `<|image_pad|>` runs sit.
public struct Qwen3VLMultimodalLayout {
    /// (3, L) int32 — (t, h, w) position of every token.
    public let positions: MLXArray
    /// One (start, count) per image, in presentation order; counts sum to the merged token total.
    public let imageRuns: [(start: Int, count: Int)]

    /// Port of transformers' `Qwen3VLModel.get_rope_index` for a single unpadded sequence.
    /// - Parameters:
    ///   - mmTokenTypes: 0 for text, 1 for `<|image_pad|>` — the processor convention; the
    ///     vision start/end delimiters are *text* here even though H3 tags them as video rows.
    ///   - imageGrids: per image, the (gridH, gridW) patch grid BEFORE spatial merge.
    public init(mmTokenTypes: [Int32], imageGrids: [(h: Int, w: Int)], mergeSize: Int = 2) throws {
        var t = [Int32](), h = [Int32](), w = [Int32]()
        t.reserveCapacity(mmTokenTypes.count)
        h.reserveCapacity(mmTokenTypes.count)
        w.reserveCapacity(mmTokenTypes.count)
        var runs = [(start: Int, count: Int)]()
        var currentPosition: Int32 = 0
        var nextImage = 0
        var index = 0
        while index < mmTokenTypes.count {
            var end = index
            while end < mmTokenTypes.count && mmTokenTypes[end] == mmTokenTypes[index] { end += 1 }
            if mmTokenTypes[index] == 0 {
                for offset in 0..<Int32(end - index) {
                    t.append(currentPosition + offset)
                    h.append(currentPosition + offset)
                    w.append(currentPosition + offset)
                }
                currentPosition += Int32(end - index)
            } else {
                guard nextImage < imageGrids.count else {
                    throw H3Error.invalidInput("More image-token runs than image grids.")
                }
                let grid = imageGrids[nextImage]
                nextImage += 1
                let (mergedH, mergedW) = (grid.h / mergeSize, grid.w / mergeSize)
                guard end - index == mergedH * mergedW else {
                    throw H3Error.invalidInput(
                        "Image run of \(end - index) tokens does not match grid "
                            + "\(grid.h)x\(grid.w) (\(mergedH * mergedW) merged tokens).")
                }
                runs.append((index, end - index))
                for row in 0..<mergedH {
                    for column in 0..<mergedW {
                        t.append(currentPosition)
                        h.append(currentPosition + Int32(row))
                        w.append(currentPosition + Int32(column))
                    }
                }
                // The clock advances by the larger merged side, not the token count.
                currentPosition += Int32(max(mergedH, mergedW))
            }
            index = end
        }
        guard nextImage == imageGrids.count else {
            throw H3Error.invalidInput("\(imageGrids.count - nextImage) image grids without token runs.")
        }
        positions = MLXArray(t + h + w, [3, mmTokenTypes.count])
        imageRuns = runs
    }
}

// MARK: - Conditioner

/// The truncated Qwen3-VL text stack: embeddings + `numLayers` decoder layers, no final norm.
/// Its output IS MiniMax-H3's text conditioning.
public final class Qwen3VLTextEncoder: Module {
    public let config: Qwen3VLTextConfig
    public let numLayers: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3VLDecoderLayer]

    /// - Parameter numLayers: decoder layers to materialize; H3 needs `hidden_states[50]`,
    ///   i.e. the output of layer index 49, so 50 layers.
    public init(config: Qwen3VLTextConfig, numLayers: Int = H3Constants.textEncoderLayer) {
        precondition(numLayers <= config.numHiddenLayers)
        self.config = config
        self.numLayers = numLayers
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0..<numLayers).map { _ in Qwen3VLDecoderLayer(config: config) }
    }

    /// tokenIds (1, L) -> unnormalized hidden states (1, L, hidden) after the last kept layer.
    ///
    /// For fl2va, `imageEmbeds` (Nv, hidden — the vision tower's merged tokens, all images
    /// concatenated in presentation order) replace the `<|image_pad|>` embeddings, and
    /// `deepstack[i]` (same layout) is added at those positions after decoder layer i.
    public func callAsFunction(
        _ tokenIds: MLXArray,
        imageEmbeds: MLXArray? = nil,
        deepstack: [MLXArray] = [],
        layout: Qwen3VLMultimodalLayout? = nil
    ) -> MLXArray {
        var x = embedTokens(tokenIds).squeezed(axis: 0)  // (L, hidden)
        if let imageEmbeds, let layout {
            var offset = 0
            for run in layout.imageRuns {
                x[run.start ..< (run.start + run.count)] =
                    imageEmbeds[offset ..< (offset + run.count)].asType(x.dtype)
                offset += run.count
            }
        }
        let (cos, sin) = mropeTable(
            positions: layout?.positions, sequenceLength: tokenIds.dim(1))
        x = x.expandedDimensions(axis: 0)
        for (index, layer) in layers.enumerated() {
            x = layer(x, cos: cos, sin: sin)
            if index < deepstack.count, let layout {
                var offset = 0
                for run in layout.imageRuns {
                    let range = run.start ..< (run.start + run.count)
                    x[0..., range] =
                        x[0..., range] + deepstack[index][offset ..< (offset + run.count)]
                        .asType(x.dtype).expandedDimensions(axis: 0)
                    offset += run.count
                }
            }
        }
        return x
    }

    /// Interleaved-mrope table in float32: per-axis angles `position · invFreq`, merged with the
    /// [24, 20, 20] channel partition (t base; h at 1,4,...,58; w at 2,5,...,59). Without a
    /// layout all three axes carry 0..<L, which reduces to standard 1-D RoPE.
    private func mropeTable(
        positions: MLXArray?, sequenceLength: Int
    ) -> (MLXArray, MLXArray) {
        let half = config.headDim / 2
        let exponents = (0..<half).map { Float(2 * $0) / Float(config.headDim) }
        let invFreq = 1.0 / pow(MLXArray(config.ropeTheta), MLXArray(exponents))  // (half,)

        guard let positions else {
            let text = MLXArray(0..<sequenceLength).asType(.float32).expandedDimensions(axis: -1)
            let freqs = text * invFreq  // (L, half)
            return (cos(freqs), sin(freqs))
        }
        let perAxis = positions.asType(.float32).expandedDimensions(axis: -1) * invFreq  // (3, L, half)
        var owner = [Int](repeating: 0, count: half)  // t owns everything not claimed below
        for channel in stride(from: 1, to: 3 * 20, by: 3) { owner[channel] = 1 }
        for channel in stride(from: 2, to: 3 * 20, by: 3) { owner[channel] = 2 }
        let masks = (0...2).map { axis in
            MLXArray(owner.map { Float($0 == axis ? 1 : 0) })
        }
        let merged = perAxis[0] * masks[0] + perAxis[1] * masks[1] + perAxis[2] * masks[2]
        return (cos(merged), sin(merged))
    }
}
