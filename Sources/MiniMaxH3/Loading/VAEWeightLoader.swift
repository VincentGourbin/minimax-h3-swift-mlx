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

    public static func loadVideoVAE(modelDirectory: URL) throws -> H3VideoVAE {
        let directory = modelDirectory.appendingPathComponent("vae")
        let config = try H3VideoVAEConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = H3VideoVAE(config: config)

        var weights = try loadShardedWeights(directory: directory) { key in
            guard key.hasPrefix("decoder.") || key.hasPrefix("post_quant_conv.") else { return nil }
            return key
                .replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
                .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj.")
                .replacingOccurrences(of: ".ff.net.2.", with: ".ff.out.")
        }
        // The released decode recipe is float16 autocast over the float32 checkpoint (the
        // published frames are the fp16 ones). fp32 decode measured 6x slower with the GPU at
        // 17%. Norm math stays fp32 — the decoder blocks upcast explicitly before every norm.
        for (key, value) in weights where value.dtype == .float32 {
            weights[key] = value.asType(.float16)
        }
        if let pqw = weights["post_quant_conv.weight"] {
            weights["post_quant_conv.weight"] = pqw.reshaped(pqw.dim(0), pqw.dim(1))
        }
        try apply(weights: weights, to: model, component: "video_vae")
        return model
    }

    // MARK: - Audio VAE (decode path)

    public static func loadAudioVAE(modelDirectory: URL) throws -> H3AudioVAE {
        let directory = modelDirectory.appendingPathComponent("audio_vae")
        let config = try H3AudioVAEConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = H3AudioVAE(config: config)

        let raw = try loadShardedWeights(directory: directory) { key in
            (key.hasPrefix("decoder.") || key.hasPrefix("dec_in_proj.")) ? key : nil
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
