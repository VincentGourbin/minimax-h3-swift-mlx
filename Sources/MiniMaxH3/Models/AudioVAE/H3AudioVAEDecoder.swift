// H3AudioVAEDecoder.swift - Decode path of AutoencoderKLMiniMaxH3Audio (BigVGAN, mono 32 kHz)
// Copyright 2026 Vincent Gourbin
//
// Milestone-1 scope: DECODE ONLY (the DAC encoder + pre_block are only needed to encode reference
// audio, i.e. ref2va). Everything runs float32 — the checkpoint degrades audibly under bf16.
// Weight-normalized convolutions are resolved at load time (weight = g * v / ||v||, torch dim=0
// semantics) so runtime modules are plain convolutions. The Kaiser-sinc resampling filters are
// persistent checkpoint buffers and are loaded, not recomputed.
//
// Stereo is carried as batch: decode([2, 32, T]) -> [2, 1, T * 800].

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct H3AudioVAEConfig: Codable, Sendable {
    public var latentDim = 2048
    public var latentChannels = 32
    public var decoderDim = 1024
    public var decoderRates = [5, 5, 2, 2, 2, 2, 2]
    public var decoderKernelSizes = [9, 9, 4, 4, 4, 4, 4]
    public var resblockKernelSizes = [3, 7, 11]
    public var resblockDilationSizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var samplingRate = 32_000
    public var latentsMean: [Float] = []
    public var latentsStd: [Float] = []

    enum CodingKeys: String, CodingKey {
        case latentDim = "latent_dim"
        case latentChannels = "latent_channels"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case decoderKernelSizes = "decoder_kernel_sizes"
        case resblockKernelSizes = "resblock_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case samplingRate = "sampling_rate"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
    }

    public init() {}

    public static func load(from url: URL) throws -> H3AudioVAEConfig {
        try JSONDecoder().decode(H3AudioVAEConfig.self, from: Data(contentsOf: url))
    }
}

// MARK: - Padding helper

/// Replicate-pad the length axis of an NLC tensor.
private func replicatePadNLC(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    var pieces = [MLXArray]()
    if left > 0 {
        let first = x[0..., ..<1, 0...]
        pieces.append(concatenated([MLXArray](repeating: first, count: left), axis: 1))
    }
    pieces.append(x)
    if right > 0 {
        let last = x[0..., (x.dim(1) - 1)..., 0...]
        pieces.append(concatenated([MLXArray](repeating: last, count: right), axis: 1))
    }
    return pieces.count == 1 ? x : concatenated(pieces, axis: 1)
}

// MARK: - Snake activations

/// BigVGAN SnakeBeta: x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2, alpha/beta stored in log
/// space as (channels,) vectors. Operates on NLC.
final class H3AudioSnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaExp = exp(alpha).reshaped(1, 1, -1)
        let betaExp = exp(beta).reshaped(1, 1, -1)
        let sine = sin(alphaExp * x)
        return x + (1.0 / (betaExp + 1e-9)) * sine * sine
    }
}

// MARK: - Anti-aliased resamplers (depthwise Kaiser-sinc, shared filter across channels)

/// 2x anti-aliased upsampler: replicate pad, transposed depthwise conv with the Kaiser filter
/// (shared across channels via a batch fold), scale by ratio, trim.
final class H3AudioUpSample1d: Module {
    let ratio: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int
    @ParameterInfo(key: "filter") var filter: MLXArray  // (1, 1, K) checkpoint buffer

    init(ratio: Int = 2, kernelSize: Int = 12) {
        self.ratio = ratio
        pad = kernelSize / ratio - 1
        padLeft = pad * ratio + (kernelSize - ratio) / 2
        padRight = pad * ratio + (kernelSize - ratio + 1) / 2
        _filter.wrappedValue = MLXArray.zeros([1, 1, kernelSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x is NLC; fold channels into batch so the shared filter is a single-channel conv.
        let (batch, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
        var folded = x.transposed(0, 2, 1).reshaped(batch * channels, length, 1)
        folded = replicatePadNLC(folded, left: pad, right: pad)
        let weight = filter.reshaped(1, filter.dim(-1), 1)  // (C_out=1, K, C_in=1)
        var out = convTransposed1d(folded, weight, stride: ratio, padding: 0)
        out = Float(ratio) * out
        out = out[0..., padLeft..<(out.dim(1) - padRight), 0...]
        return out.reshaped(batch, channels, out.dim(1)).transposed(0, 2, 1)
    }
}

/// Anti-aliased downsampler: replicate pad then strided depthwise conv with the Kaiser filter.
final class H3AudioDownSample1d: Module {
    let stride: Int
    let padLeft: Int
    let padRight: Int

    final class LowPass: Module {
        @ParameterInfo(key: "filter") var filter: MLXArray
        init(kernelSize: Int) { _filter.wrappedValue = MLXArray.zeros([1, 1, kernelSize]) }
    }

    @ModuleInfo(key: "lowpass") var lowpass: LowPass

    init(ratio: Int = 2, kernelSize: Int = 12) {
        stride = ratio
        let even = kernelSize % 2 == 0
        padLeft = kernelSize / 2 - (even ? 1 : 0)
        padRight = kernelSize / 2
        _lowpass.wrappedValue = LowPass(kernelSize: kernelSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
        var folded = x.transposed(0, 2, 1).reshaped(batch * channels, length, 1)
        folded = replicatePadNLC(folded, left: padLeft, right: padRight)
        let weight = lowpass.filter.reshaped(1, lowpass.filter.dim(-1), 1)
        let out = conv1d(folded, weight, stride: stride, padding: 0)
        return out.reshaped(batch, channels, out.dim(1)).transposed(0, 2, 1)
    }
}

/// Upsample -> SnakeBeta -> downsample: BigVGAN's alias-free activation wrapper.
final class H3AudioActivation1d: Module {
    @ModuleInfo(key: "act") var act: H3AudioSnakeBeta
    @ModuleInfo(key: "upsample") var upsample: H3AudioUpSample1d
    @ModuleInfo(key: "downsample") var downsample: H3AudioDownSample1d

    init(channels: Int) {
        _act.wrappedValue = H3AudioSnakeBeta(channels: channels)
        _upsample.wrappedValue = H3AudioUpSample1d()
        _downsample.wrappedValue = H3AudioDownSample1d()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(act(upsample(x)))
    }
}

// MARK: - BigVGAN blocks

/// Plain conv1d over NLC with pre-resolved (weight-norm merged) weights in MLX layout (C_out, K, C_in).
final class H3AudioConv1d: Module {
    let stride: Int
    let padding: Int
    let dilation: Int
    let hasBias: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0,
         dilation: Int = 1, bias: Bool = true) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        hasBias = bias
        _weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        _bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        if let bias { out = out + bias }
        return out
    }
}

/// Transposed conv1d over NLC, weights pre-resolved in MLX layout (C_out, K, C_in).
final class H3AudioConvTransposed1d: Module {
    let stride: Int
    let padding: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.stride = stride
        self.padding = padding
        _weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convTransposed1d(x, weight, stride: stride, padding: padding) + bias
    }
}

/// BigVGAN AMP block: per dilation, alias-free SnakeBeta -> dilated conv -> alias-free SnakeBeta
/// -> dilation-1 conv, residual.
final class H3AudioAMPBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [H3AudioConv1d]
    @ModuleInfo(key: "convs2") var convs2: [H3AudioConv1d]
    @ModuleInfo(key: "activations") var activations: [H3AudioActivation1d]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        _convs1.wrappedValue = dilations.map {
            H3AudioConv1d(
                inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                padding: (kernelSize * $0 - $0) / 2, dilation: $0
            )
        }
        _convs2.wrappedValue = dilations.map { _ in
            H3AudioConv1d(
                inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                padding: (kernelSize - 1) / 2, dilation: 1
            )
        }
        _activations.wrappedValue = (0..<(2 * dilations.count)).map { _ in
            H3AudioActivation1d(channels: channels)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for index in convs1.indices {
            var residual = convs1[index](activations[2 * index](x))
            residual = convs2[index](activations[2 * index + 1](residual))
            x = x + residual
        }
        return x
    }
}

// MARK: - Decoder

/// BigVGAN: (B, T, latent_dim) NLC -> (B, T * 800, 1) waveform in [-1, 1].
final class H3AudioBigVGANDecoder: Module {
    let numKernels: Int
    let numUpsamples: Int

    @ModuleInfo(key: "conv_pre") var convPre: H3AudioConv1d
    @ModuleInfo(key: "ups") var ups: [H3AudioConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [H3AudioAMPBlock]
    @ModuleInfo(key: "activation_post") var activationPost: H3AudioActivation1d
    @ModuleInfo(key: "conv_post") var convPost: H3AudioConv1d

    init(config: H3AudioVAEConfig) {
        numKernels = config.resblockKernelSizes.count
        numUpsamples = config.decoderRates.count

        _convPre.wrappedValue = H3AudioConv1d(
            inChannels: config.latentDim, outChannels: config.decoderDim, kernelSize: 7, padding: 3
        )

        var upsList = [H3AudioConvTransposed1d]()
        var resList = [H3AudioAMPBlock]()
        var channels = config.decoderDim
        for (index, (rate, kernel)) in zip(config.decoderRates, config.decoderKernelSizes).enumerated() {
            _ = index
            upsList.append(
                H3AudioConvTransposed1d(
                    inChannels: channels, outChannels: channels / 2, kernelSize: kernel,
                    stride: rate, padding: (kernel - rate) / 2
                )
            )
            channels /= 2
            for (resKernel, dilations) in zip(config.resblockKernelSizes, config.resblockDilationSizes) {
                resList.append(H3AudioAMPBlock(channels: channels, kernelSize: resKernel, dilations: dilations))
            }
        }
        _ups.wrappedValue = upsList
        _resblocks.wrappedValue = resList
        _activationPost.wrappedValue = H3AudioActivation1d(channels: channels)
        _convPost.wrappedValue = H3AudioConv1d(
            inChannels: channels, outChannels: 1, kernelSize: 7, padding: 3, bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convPre(x)
        for stage in 0..<numUpsamples {
            x = ups[stage](x)
            var summed: MLXArray? = nil
            for kernel in 0..<numKernels {
                let block = resblocks[stage * numKernels + kernel](x)
                summed = summed.map { $0 + block } ?? block
            }
            x = summed! / Float(numKernels)
            eval(x)
        }
        x = activationPost(x)
        x = convPost(x)
        return clip(x, min: -1.0, max: 1.0)
    }
}

// MARK: - Autoencoder (decode path)

public final class H3AudioVAE: Module {
    public let config: H3AudioVAEConfig

    @ModuleInfo(key: "dec_in_proj") var decInProj: H3AudioConv1d  // k=1
    @ModuleInfo(key: "decoder") var decoder: H3AudioBigVGANDecoder

    public init(config: H3AudioVAEConfig) {
        self.config = config
        _decInProj.wrappedValue = H3AudioConv1d(
            inChannels: config.latentChannels, outChannels: config.latentDim, kernelSize: 1
        )
        _decoder.wrappedValue = H3AudioBigVGANDecoder(config: config)
    }

    /// latents (B, latentChannels, T) [torch layout, denormalized] -> waveform (B, samples) in [-1, 1].
    public func decode(_ latents: MLXArray) -> MLXArray {
        let x = latents.asType(.float32).transposed(0, 2, 1)  // NLC
        let decoded = decoder(decInProj(x))  // (B, samples, 1)
        return decoded.squeezed(axis: -1)
    }
}
