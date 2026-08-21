// H3AudioVAEEncoder.swift - Encode path of AutoencoderKLMiniMaxH3Audio (DAC trunk + attention head)
// Copyright 2026 Vincent Gourbin
//
// Needed by ref2va, which conditions on reference soundtracks: waveform -> latent posterior.
// `[B, 1, samples]` (mono; MiniMax-H3 passes a stereo clip as B = 2) -> `[B, 32, samples / 800]`.
//
// Everything is float32 — the same reason the decoder is: this stack degrades audibly under bf16.
// Weight-normalized convolutions are merged at load time like the decoder's, so runtime modules are
// plain convolutions.
//
// The module keys reproduce the checkpoint's `nn.Sequential` numbering literally (`encoder.block.1.
// block.0.block.1.weight`), which is why the containers below key their children "0".."7" rather
// than by name: the alternative is string surgery in the loader, and the numbering IS the contract.

import Foundation
import MLX
import MLXNN

// MARK: - Sequential building block

/// Common base of the DAC encoder's `nn.Sequential` children.
///
/// The checkpoint keys ARE the sequential indices (`encoder.block.1.block.0.block.1.weight`), so
/// every container below holds a real Swift array — MLXNN unflattens contiguous numeric keys into
/// an array, and a struct with `"0"`-keyed properties would not survive `update(parameters:)`.
/// A shared class is what makes those arrays homogeneous.
class H3AudioOp: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

/// Run a sequential chain.
private func sequential(_ ops: [H3AudioOp], _ x: MLXArray) -> MLXArray {
    ops.reduce(x) { $1($0) }
}

// MARK: - Snake

/// DAC Snake: `x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`, `alpha` a `(1, C, 1)` checkpoint tensor.
/// Runs over NLC, so `alpha` is reshaped per call rather than stored transposed — the parameter
/// shape has to keep matching the checkpoint.
final class H3AudioSnake1d: H3AudioOp {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([1, channels, 1])
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = alpha.reshaped(1, 1, -1)
        let sine = sin(a * x)
        return x + (1.0 / (a + 1e-9)) * sine * sine
    }
}

// MARK: - DAC residual unit

/// `Snake -> dilated Conv1d(k=7) -> Snake -> Conv1d(k=1)`, residual. The shortcut is centre-cropped
/// when the dilated convolution shrinks the time axis (it does not at the released paddings, but
/// the reference crops unconditionally and a fine-tune could need it).
final class H3AudioResidualUnit: H3AudioOp {
    @ModuleInfo(key: "block") var block: [H3AudioOp]

    init(dim: Int, dilation: Int) {
        _block.wrappedValue = [
            H3AudioSnake1d(channels: dim),
            H3AudioConv1d(
                inChannels: dim, outChannels: dim, kernelSize: 7,
                padding: ((7 - 1) * dilation) / 2, dilation: dilation),
            H3AudioSnake1d(channels: dim),
            H3AudioConv1d(inChannels: dim, outChannels: dim, kernelSize: 1),
        ]
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = sequential(block, x)
        let pad = (x.dim(1) - residual.dim(1)) / 2
        let shortcut = pad > 0 ? x[0..., pad..<(x.dim(1) - pad), 0...] : x
        return shortcut + residual
    }
}

// MARK: - Encoder stage

/// Three residual units at dilations 1/3/9, a Snake, then a strided channel-doubling convolution.
final class H3AudioEncoderBlock: H3AudioOp {
    @ModuleInfo(key: "block") var block: [H3AudioOp]

    init(dim: Int, stride: Int) {
        _block.wrappedValue = [
            H3AudioResidualUnit(dim: dim / 2, dilation: 1),
            H3AudioResidualUnit(dim: dim / 2, dilation: 3),
            H3AudioResidualUnit(dim: dim / 2, dilation: 9),
            H3AudioSnake1d(channels: dim / 2),
            H3AudioConv1d(
                inChannels: dim / 2, outChannels: dim, kernelSize: 2 * stride, stride: stride,
                padding: (stride + 1) / 2),  // math.ceil(stride / 2)
        ]
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray { sequential(block, x) }
}

// MARK: - DAC trunk

/// `[B, samples, 1]` NLC -> `[B, samples / 800, latentDim]`.
final class H3AudioEncoder: H3AudioOp {
    @ModuleInfo(key: "block") var block: [H3AudioOp]

    init(config: H3AudioVAEConfig) {
        var dim = config.encoderDim
        var ops: [H3AudioOp] = [
            H3AudioConv1d(inChannels: 1, outChannels: dim, kernelSize: 7, padding: 3)
        ]
        for stride in config.encoderRates {
            dim *= 2
            ops.append(H3AudioEncoderBlock(dim: dim, stride: stride))
        }
        ops.append(H3AudioSnake1d(channels: dim))
        ops.append(
            H3AudioConv1d(
                inChannels: dim, outChannels: config.latentDim, kernelSize: 3, padding: 1))
        _block.wrappedValue = ops
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray { sequential(block, x) }
}

// MARK: - Causal attention projection (`pre_block`)

/// Causal self-attention that narrows `latentDim` (2048) to `latentChannels` (32).
///
/// Three things are unusual and all three are checkpoint contract: the QKV bias is assembled from
/// `q_bias`, a frozen zero key bias, and `v_bias`; the heads are **mean-pooled away** instead of
/// concatenated; and the head dimension that remains is average-pooled down to `outDim`.
final class H3AudioCausalAttention: Module {
    let numHeads: Int
    let headDim: Int
    let outDim: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ParameterInfo(key: "q_bias") var qBias: MLXArray
    @ParameterInfo(key: "v_bias") var vBias: MLXArray
    @ParameterInfo(key: "zero_k_bias") var zeroKBias: MLXArray
    @ModuleInfo(key: "proj") var proj: Linear

    init(inDim: Int, outDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.outDim = outDim
        headDim = inDim / numHeads
        _qkv.wrappedValue = Linear(inDim, inDim * 3, bias: false)
        _qBias.wrappedValue = MLXArray.zeros([inDim])
        _vBias.wrappedValue = MLXArray.zeros([inDim])
        _zeroKBias.wrappedValue = MLXArray.zeros([inDim])
        _proj.wrappedValue = Linear(outDim, outDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, sequence) = (x.dim(0), x.dim(1))
        let bias = concatenated([qBias, zeroKBias, vBias], axis: 0)
        let fused = qkv(x) + bias
        // (B, S, 3, H, D) -> three (B, H, S, D) for MLX's attention layout.
        let parts = fused.reshaped(batch, sequence, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let query = parts[0]
        let key = parts[1]
        let value = parts[2]

        let scale = 1.0 / Float(headDim).squareRoot()
        var out = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: scale, mask: .causal)
        // Heads are averaged away, not concatenated.
        out = out.mean(axis: 1)  // (B, S, D)
        // `F.adaptive_avg_pool1d(x, outDim)` over the head dimension. The released geometry divides
        // exactly (256 -> 32), which is plain non-overlapping mean pooling.
        precondition(headDim % outDim == 0, "adaptive pooling \(headDim) -> \(outDim) is not exact")
        out = out.reshaped(batch, sequence, outDim, headDim / outDim).mean(axis: -1)
        return proj(out)
    }
}

/// Pre-norm GeGLU MLP: `w2(gelu_tanh(w0(x)) * w1(x))`, with a LayerNorm of its own.
final class H3AudioGeGluMlp: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "w0") var w0: Linear
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(inFeatures: Int, hiddenFeatures: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: inFeatures)
        _w0.wrappedValue = Linear(inFeatures, hiddenFeatures)
        _w1.wrappedValue = Linear(inFeatures, hiddenFeatures)
        _w2.wrappedValue = Linear(hiddenFeatures, inFeatures)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = norm(x)
        return w2(geluApproximate(w0(normed)) * w1(normed))
    }
}

/// `pre_block`: the residual causal-attention + GeGLU block that rewires 2048 -> 32.
final class H3AudioAttnProjection: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: H3AudioCausalAttention
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: H3AudioGeGluMlp

    init(inDim: Int, outDim: Int, numHeads: Int, mlpRatio: Int = 2) {
        _norm1.wrappedValue = LayerNorm(dimensions: inDim)
        _attn.wrappedValue = H3AudioCausalAttention(inDim: inDim, outDim: outDim, numHeads: numHeads)
        _proj.wrappedValue = Linear(inDim, outDim)
        _norm3.wrappedValue = LayerNorm(dimensions: inDim)
        _norm2.wrappedValue = LayerNorm(dimensions: outDim)
        _mlp.wrappedValue = H3AudioGeGluMlp(inFeatures: outDim, hiddenFeatures: outDim * mlpRatio)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(norm3(x)) + attn(norm1(x))
        return projected + mlp(norm2(projected))
    }
}
