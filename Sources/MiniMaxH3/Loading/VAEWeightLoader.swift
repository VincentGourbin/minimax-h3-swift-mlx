// VAEWeightLoader.swift - Video & audio VAE checkpoint loading (decode paths)
// Copyright 2026 Vincent Gourbin
//
// Layout conversions performed here, once, at load time:
//  - post_quant_conv: torch conv3d 1x1x1 (C, C, 1, 1, 1) -> Linear (C, C)
//  - audio conv1d: torch (C_out, C_in, K) -> MLX (C_out, K, C_in)
//  - audio conv_transpose1d: torch (C_in, C_out, K) -> MLX (C_out, K, C_in)
//  - weight_norm pairs (weight_g, weight_v) -> merged weight, torch dim=0 semantics
//    (norm over every dim except 0), BEFORE the layout conversion.

import Foundation
import MLX
import MLXNN

extension H3WeightLoader {
    // MARK: - Video VAE (decode path)

    /// - Parameter includeDecoder: pass `false` for the fl2va keyframe stage, which only
    ///   encodes — it skips the ~5B-parameter ViT decoder and its ~10 GB read.
    public static func loadVideoVAE(
        modelDirectory: URL, includeEncoder: Bool = false, includeDecoder: Bool = true
    ) throws -> H3VideoVAE {
        precondition(includeEncoder || includeDecoder, "a VAE with neither half is useless")
        let directory = modelDirectory.appendingPathComponent("vae")
        let config = try H3VideoVAEConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = H3VideoVAE(
            config: config, includeEncoder: includeEncoder, includeDecoder: includeDecoder)

        var weights = try loadShardedWeights(directory: directory) { key in
            let isEncoderKey = key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.")
            let isDecoderKey = key.hasPrefix("decoder.") || key.hasPrefix("post_quant_conv.")
            guard (includeDecoder && isDecoderKey) || (includeEncoder && isEncoderKey) else {
                return nil
            }
            return key
                .replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
                .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj.")
                .replacingOccurrences(of: ".ff.net.2.", with: ".ff.out.")
        }
        if includeEncoder {
            // Encoder conv3d weights: torch (O, I, D, H, W) -> MLX NDHWC layout (O, D, H, W, I);
            // quant_conv 1x1x1 -> Linear. Encoder stays float32 (reference encodes keyframes fp32).
            for (key, value) in weights where key.hasPrefix("encoder.") && value.ndim == 5 {
                weights[key] = value.asType(.float32).transposed(0, 2, 3, 4, 1)
            }
            if let qcw = weights["quant_conv.weight"] {
                weights["quant_conv.weight"] = qcw.asType(.float32).reshaped(qcw.dim(0), qcw.dim(1))
            }
            if let qcb = weights["quant_conv.bias"] { weights["quant_conv.bias"] = qcb.asType(.float32) }
        }
        // The released decode recipe is float16 autocast over the float32 checkpoint (the
        // published frames are the fp16 ones). fp32 decode measured 6x slower with the GPU at
        // 17%. Norm math stays fp32 — the decoder blocks upcast explicitly before every norm.
        // The ENCODER stays float32 (the reference encodes keyframes without autocast).
        for (key, value) in weights
        where value.dtype == .float32 && !key.hasPrefix("encoder.") && !key.hasPrefix("quant_conv.") {
            weights[key] = value.asType(.float16)
        }
        if let pqw = weights["post_quant_conv.weight"] {
            weights["post_quant_conv.weight"] = pqw.reshaped(pqw.dim(0), pqw.dim(1))
        }
        try apply(weights: weights, to: model, component: "video_vae")
        return model
    }

    // MARK: - Audio VAE (decode path)

    /// - Parameter includeEncoder: ref2va conditions on reference soundtracks; every other path
    ///   only decodes. The whole file is 0.6 GB, so the two halves are cheap either way — the flag
    ///   is here to keep `apply`'s strict key matching honest, not to save memory.
    public static func loadAudioVAE(
        modelDirectory: URL, includeEncoder: Bool = false, includeDecoder: Bool = true
    ) throws -> H3AudioVAE {
        let directory = modelDirectory.appendingPathComponent("audio_vae")
        let config = try H3AudioVAEConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = H3AudioVAE(
            config: config, includeEncoder: includeEncoder, includeDecoder: includeDecoder)

        let raw = try loadShardedWeights(directory: directory) { key in
            let isDecoderKey = key.hasPrefix("decoder.") || key.hasPrefix("dec_in_proj.")
            let isEncoderKey = key.hasPrefix("encoder.") || key.hasPrefix("pre_block.")
                || key.hasPrefix("mean_proj.") || key.hasPrefix("logs_proj.")
            guard (includeDecoder && isDecoderKey) || (includeEncoder && isEncoderKey) else {
                return nil
            }
            return key
        }

        var weights = [String: MLXArray]()
        for (key, value) in raw {
            if key.hasSuffix(".weight_g") { continue }  // consumed with its weight_v partner
            if key.hasSuffix(".weight_v") {
                let base = String(key.dropLast(".weight_v".count))
                guard let gain = raw[base + ".weight_g"] else {
                    throw H3Error.weightLoadingFailed("audio_vae: missing weight_g for \(key)")
                }
                // weight = g * v / ||v||, norm over all dims but 0 (torch weight_norm dim=0).
                let v32 = value.asType(.float32)
                let norm = sqrt(sum(v32 * v32, axes: [1, 2], keepDims: true))
                let merged = gain.asType(.float32) * v32 / norm
                weights[audioKey(base) + ".weight"] = audioWeightLayout(base, merged)
                continue
            }
            if key.hasSuffix(".weight"), value.ndim == 3 {
                // Plain (non-normed) convs: dec_in_proj. Same layout conversion.
                let base = String(key.dropLast(".weight".count))
                weights[audioKey(base) + ".weight"] = audioWeightLayout(base, value.asType(.float32))
                continue
            }
            let renamed = key.hasSuffix(".bias")
                ? audioKey(String(key.dropLast(".bias".count))) + ".bias"
                : key
            weights[renamed] = value.asType(.float32)
        }

        try apply(weights: weights, to: model, component: "audio_vae")
        return model
    }

    /// `decoder.ups.N.0` (checkpoint's one-element ModuleList nesting) -> `decoder.ups.N`.
    private static func audioKey(_ base: String) -> String {
        base.hasSuffix(".0") && base.contains(".ups.") ? String(base.dropLast(2)) : base
    }

    /// Torch conv layouts -> MLX: conv1d (C_out, C_in, K) -> (C_out, K, C_in);
    /// conv_transpose1d (`ups.`) (C_in, C_out, K) -> (C_out, K, C_in).
    private static func audioWeightLayout(_ base: String, _ weight: MLXArray) -> MLXArray {
        base.contains(".ups.")
            ? weight.transposed(1, 2, 0)
            : weight.transposed(0, 2, 1)
    }
}
