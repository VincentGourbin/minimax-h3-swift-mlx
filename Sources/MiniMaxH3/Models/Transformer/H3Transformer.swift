// H3Transformer.swift - MiniMaxH3Transformer3DModel port (diffusers transformer_minimax_h3.py)
// Copyright 2026 Vincent Gourbin
//
// Single stack of blocks over ONE packed 1-D sequence holding text, conditioning, audio and video
// rows. Full self-attention, no cross-attention, no per-modality block weights: modality-specific
// behaviour lives only in the input projections, the per-row AdaLN modality tag and the two output
// heads. The checkpoint is guidance-distilled (one forward per step) and mixed-precision: proj_in,
// audio_proj_in, time_embedder, proj_out, audio_proj_out ship float32, everything else bfloat16.
//
// We only build padless sequences (t2va/fl2va contiguous layout: [text | cond | audio | video]),
// so no attention mask is ever needed and sequence assembly is plain concatenation — numerically
// identical to the reference's scatter into a zero buffer.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct H3TransformerConfig: Codable, Sendable {
    public var numAttentionHeads = 56
    public var attentionHeadDim = 128
    public var hiddenSize = 5376
    public var numLayers = 50
    public var numRefinerLayers = 2
    public var ffnDim = 14336
    public var inChannels = 24
    public var audioInChannels = 32
    public var patchSize = [1, 2, 2]
    public var textDim = 5120
    public var freqDim = 256
    public var timeEmbedHiddenDim = 5376
    public var timeEmbedDim = 2688
    public var ropeFreqDim = 16
    public var ropeTheta: Float = 10000.0
    public var normEps: Float = 1e-5
    public var qkNormEps: Float = 1e-5
    public var finalNormEps: Float = 1e-5

    enum CodingKeys: String, CodingKey {
        case numAttentionHeads = "num_attention_heads"
        case attentionHeadDim = "attention_head_dim"
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numRefinerLayers = "num_refiner_layers"
        case ffnDim = "ffn_dim"
        case inChannels = "in_channels"
        case audioInChannels = "audio_in_channels"
        case patchSize = "patch_size"
        case textDim = "text_dim"
        case freqDim = "freq_dim"
        case timeEmbedHiddenDim = "time_embed_hidden_dim"
        case timeEmbedDim = "time_embed_dim"
        case ropeFreqDim = "rope_freq_dim"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case qkNormEps = "qk_norm_eps"
        case finalNormEps = "final_norm_eps"
    }

    public init() {}

    public var videoPatchDim: Int { inChannels * patchSize[0] * patchSize[1] * patchSize[2] }
    /// Number of head channels the 3-axis rotary embedding rotates (2 * 3 * ropeFreqDim).
    public var rotaryDim: Int { 2 * 3 * ropeFreqDim }

    public static func load(from url: URL) throws -> H3TransformerConfig {
        try JSONDecoder().decode(H3TransformerConfig.self, from: Data(contentsOf: url))
    }
}

// MARK: - Rotary embedding

/// 3-axis rotary embedding over the (t, h, w) coordinates of the packed sequence. One shared
/// `invFreq` buffer of `ropeFreqDim` frequencies; each axis contributes that many angles, the
/// three blocks are concatenated and then doubled so rotate_half covers `2*3*ropeFreqDim` channels.
final class H3RotaryPosEmbed {
    let invFreq: MLXArray

    init(ropeFreqDim: Int, ropeTheta: Float) {
        let exponents = stride(from: 0, to: 2 * ropeFreqDim, by: 2).map {
            Float($0) / Float(2 * ropeFreqDim)
        }
        invFreq = 1.0 / pow(MLXArray(ropeTheta), MLXArray(exponents))
    }

    /// positionIds (seq, 3) float32 -> (cos, sin) each (seq, 2*3*ropeFreqDim) float32.
    func callAsFunction(_ positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let freqsPerAxis = positionIds.asType(.float32).expandedDimensions(axis: -1)
            * invFreq.reshaped(1, 1, -1)  // (seq, 3, ropeFreqDim)
        let seq = freqsPerAxis.dim(0)
        var freqs = freqsPerAxis.reshaped(seq, -1)  // (t | h | w) concatenated
        freqs = concatenated([freqs, freqs], axis: -1)
        return (cos(freqs), sin(freqs))
    }
}

/// Rotate the leading `rotaryDim` channels of every head (rotate_half convention), pass the rest.
/// x is (B, seq, heads, headDim); cos/sin are (seq, rotaryDim).
public func applyH3RotaryEmb(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let rotaryDim = cos.dim(-1)
    let xRotary = x[.ellipsis, ..<rotaryDim]
    let xPass = x[.ellipsis, rotaryDim...]

    let cosB = cos.asType(x.dtype).reshaped(1, cos.dim(0), 1, rotaryDim)
    let sinB = sin.asType(x.dtype).reshaped(1, sin.dim(0), 1, rotaryDim)
    let half = rotaryDim / 2
    let x1 = xRotary[.ellipsis, ..<half]
    let x2 = xRotary[.ellipsis, half...]
    let rotated = concatenated([-x2, x1], axis: -1)
    return concatenated([xRotary * cosB + rotated * sinB, xPass], axis: -1)
}

// MARK: - Attention

/// Full self-attention over the packed sequence, bias-free projections, per-head q/k RMSNorm.
/// Inner dim (heads * headDim = 7168) is larger than the hidden size (5376).
final class H3Attention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(hiddenSize: Int, heads: Int, headDim: Int, qkNormEps: Float) {
        self.heads = heads
        self.headDim = headDim
        let innerDim = heads * headDim
        _toQ.wrappedValue = Linear(hiddenSize, innerDim, bias: false)
        _toK.wrappedValue = Linear(hiddenSize, innerDim, bias: false)
        _toV.wrappedValue = Linear(hiddenSize, innerDim, bias: false)
        _toOut.wrappedValue = Linear(innerDim, hiddenSize, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)
    }

    /// x (B, seq, hidden); rotary optional (token refiner runs without it).
    func callAsFunction(_ x: MLXArray, rotary: (cos: MLXArray, sin: MLXArray)? = nil) -> MLXArray {
        let (batch, seq) = (x.dim(0), x.dim(1))
        var q = toQ(x).reshaped(batch, seq, heads, headDim)
        var k = toK(x).reshaped(batch, seq, heads, headDim)
        let v = toV(x).reshaped(batch, seq, heads, headDim)

        q = normQ(q)
        k = normK(k)
        if let rotary {
            q = applyH3RotaryEmb(q, cos: rotary.cos, sin: rotary.sin)
            k = applyH3RotaryEmb(k, cos: rotary.cos, sin: rotary.sin)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: nil
        )
        return toOut(out.transposed(0, 2, 1, 3).reshaped(batch, seq, heads * headDim))
    }
}

// MARK: - Feed-forward (SwiGLU, bias-free)

/// diffusers FeedForward(activation="swiglu"): proj (dim -> 2*inner), x * silu(gate), out linear.
/// Checkpoint keys `ff.net.0.proj.*` / `ff.net.2.*` are remapped to `ff.proj.*` / `ff.out.*`.
final class H3FeedForward: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(dim: Int, innerDim: Int) {
        _proj.wrappedValue = Linear(dim, innerDim * 2, bias: false)
        _out.wrappedValue = Linear(innerDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(x)
        let split = projected.split(parts: 2, axis: -1)
        return out(split[0] * silu(split[1]))
    }
}

/// Compute dtype of a possibly-quantized Linear. A QuantizedLinear's `weight` is the packed
/// uint32 tensor, and for the mxfp/nvfp modes its `scales` are uint8 exponents — neither is a
/// valid activation dtype. Only trust a floating scales dtype; otherwise the packed stream runs
/// at the checkpoint's bf16.
@inline(__always)
func computeDType(_ linear: Linear) -> DType {
    guard let quantized = linear as? QuantizedLinear else { return linear.weight.dtype }
    switch quantized.scales.dtype {
    case .float16, .bfloat16, .float32: return quantized.scales.dtype
    default: return .bfloat16
    }
}

// MARK: - Timestep embedding

/// Sinusoidal embedding (flip_sin_to_cos=true, downscale_freq_shift=0) + MLP 256 -> 5376 -> 2688.
/// Timesteps are consumed unscaled in [0, 1]; the whole path runs float32 in the checkpoint.
final class H3TimeEmbedder: Module {
    let freqDim: Int
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(freqDim: Int, hiddenDim: Int, outDim: Int) {
        self.freqDim = freqDim
        _linear1.wrappedValue = Linear(freqDim, hiddenDim, bias: true)
        _linear2.wrappedValue = Linear(hiddenDim, outDim, bias: true)
    }

    /// timesteps (T,) float32 -> (T, timeEmbedDim) float32.
    func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        let half = freqDim / 2
        let exponent = MLXArray(0..<half).asType(.float32) * (-logf(10000.0) / Float(half))
        let angles = timesteps.asType(.float32).expandedDimensions(axis: -1) * exp(exponent)
        // flip_sin_to_cos: [cos | sin]
        let embedding = concatenated([cos(angles), sin(angles)], axis: -1)
        return linear2(silu(linear1(embedding.asType(computeDType(linear1)))))
    }
}

// MARK: - AdaLN modulation

/// Projects the shared timestep embedding into the six per-(timestep, modality) modulation tensors
/// of one block. Rows laid out [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...], addressed by
/// `timestepIndex * 3 + tokenTag`. SiLU runs at temb's precision (float32) and only its result is
/// cast to the bfloat16 projection — the rounding order is part of the checkpoint contract.
final class H3AdaLNModulation: Module {
    static let modalityCount = 3
    let hiddenSize: Int
    @ModuleInfo(key: "linear") var linear: Linear

    init(timeEmbedDim: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        _linear.wrappedValue = Linear(
            timeEmbedDim, 6 * hiddenSize * Self.modalityCount, bias: true
        )
    }

    /// temb (T, timeEmbedDim) fp32 -> six tensors (T * 3, hiddenSize).
    func callAsFunction(_ temb: MLXArray) -> [MLXArray] {
        let projected = linear(silu(temb).asType(computeDType(linear)))
        return projected.reshaped(-1, 6 * hiddenSize).split(parts: 6, axis: -1)
    }
}

/// Final norm: RMSNorm + shift/scale from temb, addressed per row by timestep index.
/// Projection halves are SHIFT then SCALE (Wan/LTX2 output-layer order).
final class H3AdaLNOut: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int, timeEmbedDim: Int, eps: Float) {
        _norm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        _linear.wrappedValue = Linear(timeEmbedDim, 2 * hiddenSize, bias: true)
    }

    /// x (B, seq, hidden); temb (T, timeEmbedDim) fp32; timestepIndices (seq,) int32.
    func callAsFunction(_ x: MLXArray, temb: MLXArray, timestepIndices: MLXArray) -> MLXArray {
        let projected = linear(silu(temb).asType(computeDType(linear)))
        let split = projected.split(parts: 2, axis: -1)
        let shift = split[0].take(timestepIndices, axis: 0)
        let scale = split[1].take(timestepIndices, axis: 0)
        return norm(x) * (1.0 + scale) + shift
    }
}

// MARK: - Blocks

/// Plain pre-norm transformer block refining the projected text stream. No AdaLN, no rotary.
final class H3TokenRefinerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attn: H3Attention
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "ff") var ff: H3FeedForward

    init(config: H3TransformerConfig) {
        _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _attn.wrappedValue = H3Attention(
            hiddenSize: config.hiddenSize, heads: config.numAttentionHeads,
            headDim: config.attentionHeadDim, qkNormEps: config.qkNormEps
        )
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _ff.wrappedValue = H3FeedForward(dim: config.hiddenSize, innerDim: config.ffnDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x + attn(norm1(x))
        x = x + ff(norm2(x))
        return x
    }
}

final class H3TokenRefiner: Module {
    @ModuleInfo(key: "refiner_blocks") var blocks: [H3TokenRefinerBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm

    init(config: H3TransformerConfig) {
        _blocks.wrappedValue = (0..<config.numRefinerLayers).map { _ in H3TokenRefinerBlock(config: config) }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.finalNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for block in blocks { x = block(x) }
        return finalNorm(x)
    }
}

/// H3 block: pre-norm attention and feed-forward, each modulated by AdaLN parameters selected
/// per row from the (timestep, modality) table.
final class H3TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attn: H3Attention
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "ff") var ff: H3FeedForward
    @ModuleInfo(key: "adaln_proj") var adalnProj: H3AdaLNModulation

    init(config: H3TransformerConfig) {
        _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _attn.wrappedValue = H3Attention(
            hiddenSize: config.hiddenSize, heads: config.numAttentionHeads,
            headDim: config.attentionHeadDim, qkNormEps: config.qkNormEps
        )
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _ff.wrappedValue = H3FeedForward(dim: config.hiddenSize, innerDim: config.ffnDim)
        _adalnProj.wrappedValue = H3AdaLNModulation(
            timeEmbedDim: config.timeEmbedDim, hiddenSize: config.hiddenSize
        )
    }

    /// Graph-compiled forward, built lazily on first use. Fuses the elementwise glue (AdaLN
    /// affines, gates, residuals, SwiGLU pointwise) into fewer kernels; the math is unchanged
    /// and the weights are captured as constants. Compilation happens once per input signature
    /// — the first denoising step absorbs it, like the existing kernel-compile spike.
    private var compiledForward: (@Sendable ([MLXArray]) -> [MLXArray])?

    /// adalnIndices = timestepIndices * 3 + tokenTags, (seq,) int32.
    func callAsFunction(
        _ x: MLXArray,
        temb: MLXArray,
        adalnIndices: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray),
        compiled: Bool = false
    ) -> MLXArray {
        guard compiled else {
            return forward(x, temb: temb, adalnIndices: adalnIndices, rotary: rotary)
        }
        if compiledForward == nil {
            compiledForward = MLX.compile { [unowned self] arrays in
                [forward(arrays[0], temb: arrays[1], adalnIndices: arrays[2],
                         rotary: (arrays[3], arrays[4]))]
            }
        }
        return compiledForward!([x, temb, adalnIndices, rotary.cos, rotary.sin])[0]
    }

    private func forward(
        _ x: MLXArray,
        temb: MLXArray,
        adalnIndices: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let modulation = adalnProj(temb)
        let (shiftMsa, scaleMsa, gateMsa) = (
            modulation[0].take(adalnIndices, axis: 0),
            modulation[1].take(adalnIndices, axis: 0),
            modulation[2].take(adalnIndices, axis: 0)
        )
        let (shiftMlp, scaleMlp, gateMlp) = (
            modulation[3].take(adalnIndices, axis: 0),
            modulation[4].take(adalnIndices, axis: 0),
            modulation[5].take(adalnIndices, axis: 0)
        )

        var x = x
        let attnInput = norm1(x) * (1.0 + scaleMsa) + shiftMsa
        x = x + gateMsa * attn(attnInput, rotary: rotary)
        let ffInput = norm2(x) * (1.0 + scaleMlp) + shiftMlp
        x = x + gateMlp * ff(ffInput)
        return x
    }
}

// MARK: - Model

public final class H3Transformer: Module {
    public let config: H3TransformerConfig
    let rope: H3RotaryPosEmbed

    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "audio_proj_in") var audioProjIn: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: H3TimeEmbedder
    @ModuleInfo(key: "token_refiner") var tokenRefiner: H3TokenRefiner
    @ModuleInfo(key: "transformer_blocks") var blocks: [H3TransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: H3AdaLNOut
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear

    /// Compile each block's forward into a fused graph (`MLX.compile`). Same math, fewer
    /// kernels — recovers the GPU idle share between elementwise ops, which the cost map
    /// measured at ~26 % of denoising at 9k tokens and ~8 % at 24k. Off by default until a
    /// generation has validated it on the caller's side; the parity harness covers numerics.
    public var compileBlocks = false

    public init(config: H3TransformerConfig) {
        self.config = config
        rope = H3RotaryPosEmbed(ropeFreqDim: config.ropeFreqDim, ropeTheta: config.ropeTheta)

        _projIn.wrappedValue = Linear(config.videoPatchDim, config.hiddenSize, bias: true)
        _audioProjIn.wrappedValue = Linear(config.audioInChannels, config.hiddenSize, bias: true)
        _contextEmbedder.wrappedValue = Linear(config.textDim, config.hiddenSize, bias: true)
        _timeEmbedder.wrappedValue = H3TimeEmbedder(
            freqDim: config.freqDim, hiddenDim: config.timeEmbedHiddenDim, outDim: config.timeEmbedDim
        )
        _tokenRefiner.wrappedValue = H3TokenRefiner(config: config)
        _blocks.wrappedValue = (0..<config.numLayers).map { _ in H3TransformerBlock(config: config) }
        _normOut.wrappedValue = H3AdaLNOut(
            hiddenSize: config.hiddenSize, timeEmbedDim: config.timeEmbedDim, eps: config.finalNormEps
        )
        _projOut.wrappedValue = Linear(config.hiddenSize, config.videoPatchDim, bias: true)
        _audioProjOut.wrappedValue = Linear(config.hiddenSize, config.audioInChannels, bias: true)
    }

    /// One forward over the packed sequence (velocity prediction for every row at once).
    ///
    /// - Parameters:
    ///   - videoRows: (numVideoRows, videoPatchDim) — conditioning rows first, then targets.
    ///   - audioRows: (numAudioRows, audioInChannels) — channel-major.
    ///   - textEmbeds: (1, numTextTokens, textDim).
    ///   - timesteps: distinct timestep values present in the sequence, sorted, in [0, 1].
    ///   - timestepIndices: (seq,) index of every row's timestep into `timesteps`.
    ///   - layout: the packed layout (contiguous [text | cond | audio | video] — padless).
    /// - Returns: video velocity (numVideoRows, videoPatchDim) and audio velocity
    ///   (numAudioRows, audioInChannels), float32.
    public func callAsFunction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        textEmbeds: MLXArray,
        timesteps: [Float],
        timestepIndices: MLXArray,
        layout: H3PackedSequence
    ) -> (video: MLXArray, audio: MLXArray) {
        let rotary = rope(layout.positionIds)

        // 1. Per-modality projections; the text stream sets the packed sequence's dtype (bf16).
        // proj_in / audio_proj_in are float32 modules, context_embedder is bf16.
        let videoEmbeds = projIn(videoRows.asType(computeDType(projIn)))
        let audioEmbeds = audioProjIn(audioRows.asType(computeDType(audioProjIn)))
        var textStream = contextEmbedder(textEmbeds.asType(computeDType(contextEmbedder)))
        textStream = tokenRefiner(textStream)
        let streamType = textStream.dtype

        // 2. Assemble the packed sequence. The t2va/fl2va layout is contiguous
        // [text | condition video | audio | target video], so concatenation reproduces the
        // reference's index_copy scatter exactly.
        let conditionRows = layout.numConditionVideoRows
        var pieces: [MLXArray] = [textStream[0]]
        if conditionRows > 0 { pieces.append(videoEmbeds[..<conditionRows].asType(streamType)) }
        pieces.append(audioEmbeds.asType(streamType))
        pieces.append(videoEmbeds[conditionRows...].asType(streamType))
        var x = concatenated(pieces, axis: 0).expandedDimensions(axis: 0)  // (1, seq, hidden)

        // 3. One timestep embedding per distinct noise level, float32, shared by every AdaLN.
        let temb = timeEmbedder(MLXArray(timesteps))

        // 4. Row -> AdaLN table row.
        let adalnIndices = timestepIndices * Int32(H3AdaLNModulation.modalityCount) + layout.tokenTags

        for block in blocks {
            x = block(x, temb: temb, adalnIndices: adalnIndices, rotary: rotary,
                      compiled: compileBlocks)
        }

        // 5. Output norm (indexed per row by timestep), then both float32 heads over every row;
        // rows of each modality selected after, by the contiguous layout.
        x = normOut(x, temb: temb, timestepIndices: timestepIndices)
        x = x.asType(computeDType(projOut))
        let videoOut = projOut(x)[0]
        let audioOut = audioProjOut(x)[0]

        let textCount = layout.numTextTokens
        let audioStart = textCount + conditionRows
        let videoStart = audioStart + layout.numAudioRows
        var videoPieces = [MLXArray]()
        if conditionRows > 0 { videoPieces.append(videoOut[textCount..<audioStart]) }
        videoPieces.append(videoOut[videoStart...])
        let video = videoPieces.count == 1 ? videoPieces[0] : concatenated(videoPieces, axis: 0)
        return (video, audioOut[audioStart..<videoStart])
    }
}
