// QuantizationConfig.swift - Quantization presets (flux-2-swift-mlx pattern)
// Copyright 2026 Vincent Gourbin
//
// A single `descriptor` switch wires every property of a preset so a new case cannot be
// half-implemented. Module exclusions follow MiniMax's own int8 deployment recipe (diffusers
// docs): the mixed-precision fp32 modules, the token refiner and the output norm stay at full
// precision; AdaLN projections ARE quantized.

import Foundation
import MLX
import MLXNN

public enum H3Quantization: String, CaseIterable, Sendable, Codable {
    case none
    case qint8
    case qint6
    case int4
    case mxfp8
    case mxfp4
    case nvfp4

    /// (bits, groupSize, mode) for MLXNN's quantizer, nil = keep full precision.
    /// Group sizes are format-dictated (MLX rejects others): affine 64, mxfp 32, nvfp4 16.
    public var descriptor: (bits: Int, groupSize: Int, mode: QuantizationMode)? {
        switch self {
        case .none: return nil
        case .qint8: return (8, 64, .affine)
        case .qint6: return (6, 64, .affine)
        case .int4: return (4, 64, .affine)
        case .mxfp8: return (8, 32, .mxfp8)
        case .mxfp4: return (4, 32, .mxfp4)
        case .nvfp4: return (4, 16, .nvfp4)
        }
    }

    public var displayName: String {
        guard let descriptor else { return "full precision (bf16/fp32 mixed)" }
        return "\(rawValue) (\(descriptor.bits)-bit, group \(descriptor.groupSize))"
    }

    /// Approximate resident size of the 33B transformer under this preset.
    public var estimatedTransformerGB: Double {
        switch self {
        case .none: return 61.7
        case .qint8, .mxfp8: return 18.5
        case .qint6: return 14.5
        case .int4, .mxfp4, .nvfp4: return 10.5
        }
    }
}

enum H3QuantizationFilter {
    /// Transformer modules kept at full precision (MiniMax's published int8 recipe).
    static let transformerExclusions = [
        "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
        "token_refiner", "norm_out", "proj_out", "audio_proj_out",
    ]

    /// Quantize every Linear except the excluded prefixes. Embeddings stay full precision
    /// (the text encoder recipe excludes embed_tokens; the transformer has none).
    static func apply(_ quantization: H3Quantization, to model: Module, exclusions: [String]) {
        guard let descriptor = quantization.descriptor else { return }
        quantize(model: model) { path, module in
            guard module is Linear else { return nil }
            guard !exclusions.contains(where: { path == $0 || path.hasPrefix($0 + ".") }) else {
                return nil
            }
            return (groupSize: descriptor.groupSize, bits: descriptor.bits, mode: descriptor.mode)
        }
    }
}
