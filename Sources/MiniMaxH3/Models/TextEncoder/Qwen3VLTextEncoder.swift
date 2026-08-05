// Qwen3VLTextEncoder.swift - Qwen3-VL-32B text stack as MiniMax-H3's conditioner
// Copyright 2026 Vincent Gourbin
//
// MiniMax-H3 conditions on the *unnormalized* hidden state after decoder layer 50 of its Qwen3-VL
// conditioner (hidden_states[50], with hidden_states[0] the embedding output). The final norm and
// LM head are never used, so for text-only (t2va) we materialize layers 0..<50 plus the embedding
// and skip the vision tower entirely — the last 14 layers, the norm and the head are dead weight.
//
// Text-only mrope note: Qwen3-VL's interleaved 3-axis rotary assigns each frequency channel to one
// of (t, h, w); for pure text all three axes carry the same position, which collapses to standard
// 1-D RoPE at rope_theta = 5e6. The multimodal path (vision blocks for fl2va) will need the real
// mrope layout and the vision tower — deferred to the fl2va milestone.

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
    public func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        var x = embedTokens(tokenIds)
        let (cos, sin) = ropeTable(sequenceLength: tokenIds.dim(1), dtype: x.dtype)
        for layer in layers {
            x = layer(x, cos: cos, sin: sin)
        }
        return x
    }

    /// 1-D RoPE table (text-only collapse of the interleaved mrope): positions 0..<L, theta 5e6.
    private func ropeTable(sequenceLength: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let half = config.headDim / 2
        let exponents = (0..<half).map { Float(2 * $0) / Float(config.headDim) }
        let invFreq = 1.0 / pow(MLXArray(config.ropeTheta), MLXArray(exponents))
        let positions = MLXArray(0..<sequenceLength).asType(.float32).expandedDimensions(axis: -1)
        let freqs = positions * invFreq  // (L, half)
        return (cos(freqs), sin(freqs))
    }
}
