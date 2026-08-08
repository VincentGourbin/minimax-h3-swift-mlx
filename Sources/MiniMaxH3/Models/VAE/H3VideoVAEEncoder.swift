// H3VideoVAEEncoder.swift - Causal CNN encoder of AutoencoderKLMiniMaxH3 (fl2va keyframes)
// Copyright 2026 Vincent Gourbin
//
// The encode path needed for keyframe conditioning: causal 3D CNN (spatial reflect padding,
// temporal zero padding of kernel-1 leading frames), frame-isolated GroupNorm, space/time
// strided downsamplers, quant_conv. Runs in float32 like the reference (keyframes are single
// frames — speed is irrelevant, parity is not).
//
// Implementation choices (informed by ltx-video-swift-mlx, deviating where H3 requires):
// native MLX conv3d in NDHWC throughout (H3 has temporal strides, which ltx's conv2d-loop
// pattern silently drops); manual reflect padding (MLX PadMode lacks it; correct for p=1, all
// H3 kernels are 3); GroupNorm folds T into the batch axis (per-frame statistics) with fp32
// math.

import Foundation
import MLX
import MLXNN

// MARK: - Causal conv3d (NDHWC)

/// Conv3d with symmetric spatial reflect padding and causal temporal zero padding.
/// Activations are NDHWC (B, T, H, W, C); weights stored in MLX layout (O, D, H, W, I).
final class H3CausalConv3d: Module {
    let strides: (t: Int, h: Int, w: Int)
    let spatialPadding: Int
    let temporalPadding: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        strides: (t: Int, h: Int, w: Int) = (1, 1, 1),
        spatialPadding: Int = 0,
        temporalPadding: Int = 0
    ) {
        self.strides = strides
        self.spatialPadding = spatialPadding
        self.temporalPadding = temporalPadding
        _weight.wrappedValue = MLXArray.zeros(
            [outChannels, kernelSize, kernelSize, kernelSize, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    /// Reflect-pad (p=1 semantics) both spatial axes of an NDHWC tensor.
    static func reflectPadSpatial(_ x: MLXArray, _ padding: Int) -> MLXArray {
        guard padding > 0 else { return x }
        precondition(padding == 1, "manual reflect padding implemented for p=1 (k=3) only")
        let height = x.dim(2)
        var padded = concatenated(
            [x[0..., 0..., 1..<2, 0..., 0...], x, x[0..., 0..., (height - 2)..<(height - 1), 0..., 0...]],
            axis: 2)
        let width = padded.dim(3)
        padded = concatenated(
            [padded[0..., 0..., 0..., 1..<2, 0...], padded,
             padded[0..., 0..., 0..., (width - 2)..<(width - 1), 0...]],
            axis: 3)
        return padded
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if spatialPadding > 0 {
            x = Self.reflectPadSpatial(x, spatialPadding)
        }
        if temporalPadding > 0 {
            // Causal: prepend zero frames, append nothing (reference pads with constants).
            let zeros = MLXArray.zeros(
                [x.dim(0), temporalPadding, x.dim(2), x.dim(3), x.dim(4)], dtype: x.dtype)
            x = concatenated([zeros, x], axis: 1)
        }
        return conv3d(
            x, weight, stride: [strides.t, strides.h, strides.w], padding: [0, 0, 0]
        ) + bias
    }
}

// MARK: - Frame-isolated GroupNorm

/// GroupNorm whose statistics never mix across frames: T is folded into the batch axis
/// (`use_t_isolated_gn` in the original config). NDHWC activations, fp32 statistics.
final class H3VideoGroupNorm: Module {
    let groups: Int
    let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(groups: Int, channels: Int, eps: Float) {
        self.groups = groups
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, frames, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var y = x.asType(.float32).reshaped(batch * frames, height * width, groups, channels / groups)
        let mean = y.mean(axes: [1, 3], keepDims: true)
        let centered = y - mean
        let variance = (centered * centered).mean(axes: [1, 3], keepDims: true)
        y = centered * rsqrt(variance + eps)
        y = y.reshaped(batch, frames, height, width, channels)
        return (y * weight + bias).asType(x.dtype)
    }
}

// MARK: - Blocks

final class H3VideoResnetBlock3d: Module {
    @ModuleInfo(key: "norm1") var norm1: H3VideoGroupNorm
    @ModuleInfo(key: "conv1") var conv1: H3CausalConv3d
    @ModuleInfo(key: "norm2") var norm2: H3VideoGroupNorm
    @ModuleInfo(key: "conv2") var conv2: H3CausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: H3CausalConv3d?

    init(inChannels: Int, outChannels: Int, groups: Int, eps: Float) {
        _norm1.wrappedValue = H3VideoGroupNorm(groups: groups, channels: inChannels, eps: eps)
        _conv1.wrappedValue = H3CausalConv3d(
            inChannels: inChannels, outChannels: outChannels, kernelSize: 3,
            spatialPadding: 1, temporalPadding: 2)
        _norm2.wrappedValue = H3VideoGroupNorm(groups: groups, channels: outChannels, eps: eps)
        _conv2.wrappedValue = H3CausalConv3d(
            inChannels: outChannels, outChannels: outChannels, kernelSize: 3,
            spatialPadding: 1, temporalPadding: 2)
        _convShortcut.wrappedValue = inChannels != outChannels
            ? H3CausalConv3d(inChannels: inChannels, outChannels: outChannels, kernelSize: 1)
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        if let convShortcut { residual = convShortcut(residual) }
        return residual + h
    }
}

/// Strided 3x3x3 downsampler: a spatial stride of 2 is preceded by an asymmetric bottom/right
/// reflect pad of 1, so the output is exactly ceil(size / 2).
final class H3VideoDownsample3d: Module {
    let spatialStride: Int
    @ModuleInfo(key: "conv") var conv: H3CausalConv3d

    init(channels: Int, temporalStride: Int, spatialStride: Int) {
        self.spatialStride = spatialStride
        _conv.wrappedValue = H3CausalConv3d(
            inChannels: channels, outChannels: channels, kernelSize: 3,
            strides: (temporalStride, spatialStride, spatialStride),
            spatialPadding: 0, temporalPadding: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if spatialStride == 2 {
            // Asymmetric bottom/right reflect pad (p=1 mirror of the interior neighbour).
            let height = x.dim(2)
            x = concatenated([x, x[0..., 0..., (height - 2)..<(height - 1), 0..., 0...]], axis: 2)
            let width = x.dim(3)
            x = concatenated([x, x[0..., 0..., 0..., (width - 2)..<(width - 1), 0...]], axis: 3)
        }
        return conv(x)
    }
}

final class H3VideoDownBlock3d: Module {
    @ModuleInfo(key: "resnets") var resnets: [H3VideoResnetBlock3d]
    @ModuleInfo(key: "downsamplers") var downsamplers: [H3VideoDownsample3d]?

    init(
        inChannels: Int, outChannels: Int, layers: Int,
        temporalFactor: Int, spatialFactor: Int, groups: Int, eps: Float
    ) {
        _resnets.wrappedValue = (0..<layers).map { index in
            H3VideoResnetBlock3d(
                inChannels: index == 0 ? inChannels : outChannels,
                outChannels: outChannels, groups: groups, eps: eps)
        }
        _downsamplers.wrappedValue = temporalFactor * spatialFactor > 1
            ? [H3VideoDownsample3d(
                channels: outChannels, temporalStride: temporalFactor, spatialStride: spatialFactor)]
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for resnet in resnets { x = resnet(x) }
        if let downsamplers { for downsampler in downsamplers { x = downsampler(x) } }
        return x
    }
}

// MARK: - Encoder

/// Causal 3D CNN encoder: pixels (NDHWC, ImageNet-normalized) -> moments (2 * latentChannels).
final class H3VideoEncoder3d: Module {
    @ModuleInfo(key: "conv_in") var convIn: H3CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [H3VideoDownBlock3d]
    @ModuleInfo(key: "norm_out") var normOut: H3VideoGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: H3CausalConv3d

    init(config: H3VideoVAEConfig) {
        let channels = config.blockOutChannels
        _convIn.wrappedValue = H3CausalConv3d(
            inChannels: 3, outChannels: channels[0], kernelSize: 3,
            spatialPadding: 1, temporalPadding: 2)
        var blocks = [H3VideoDownBlock3d]()
        var inChannels = channels[0]
        for level in 0..<channels.count {
            blocks.append(
                H3VideoDownBlock3d(
                    inChannels: inChannels, outChannels: channels[level],
                    layers: config.layersPerBlock,
                    temporalFactor: config.temporalDownsampleFactors[level],
                    spatialFactor: config.spatialDownsampleFactors[level],
                    groups: config.normNumGroups, eps: config.normEps))
            inChannels = channels[level]
        }
        _downBlocks.wrappedValue = blocks
        _normOut.wrappedValue = H3VideoGroupNorm(
            groups: config.normNumGroups, channels: channels.last!, eps: config.normEps)
        _convOut.wrappedValue = H3CausalConv3d(
            inChannels: channels.last!, outChannels: 2 * config.latentChannels, kernelSize: 3,
            spatialPadding: 1, temporalPadding: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)
        eval(x)
        for block in downBlocks {
            x = block(x)
            eval(x)
        }
        x = convOut(silu(normOut(x)))
        return x
    }
}
