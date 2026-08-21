// WeightLoader.swift - Sharded safetensors loading with key remapping and diagnostics
// Copyright 2026 Vincent Gourbin
//
// Loads diffusers-layout checkpoints from a local model directory. Arrays from
// `loadArrays` are lazy/memory-mapped; we
// materialize them with an explicit eval after the update so the denoising loop never faults
// pages back in from the external SSD.

import Foundation
import MLX
import MLXNN

public enum H3WeightLoader {
    // MARK: - Generic sharded loading

    /// Load every `*.safetensors` in a directory, remap keys, and return the merged dictionary.
    /// Keys for which `remap` returns nil are dropped (dead weight, e.g. truncated layers).
    static func loadShardedWeights(
        directory: URL,
        remap: (String) -> String?
    ) throws -> [String: MLXArray] {
        let shards = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else {
            throw H3Error.modelNotFound("No .safetensors shards in \(directory.path)")
        }

        var merged = [String: MLXArray]()
        for shard in shards {
            let arrays = try loadArrays(url: shard)
            for (key, value) in arrays {
                guard let mappedKey = remap(key) else { continue }
                merged[mappedKey] = value
            }
        }
        return merged
    }

    /// Apply weights to a module and report both directions of key mismatch.
    /// - Parameter allowMissing: model parameter key prefixes that may legitimately stay
    ///   at their initialized values (must be empty for H3 components — every key must load).
    static func apply(
        weights: [String: MLXArray],
        to model: Module,
        component: String,
        allowMissing: [String] = []
    ) throws {
        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let weightKeys = Set(weights.keys)

        let missing = modelKeys.subtracting(weightKeys)
            .filter { key in !allowMissing.contains { key.hasPrefix($0) } }
        let unused = weightKeys.subtracting(modelKeys)
        guard missing.isEmpty else {
            throw H3Error.weightLoadingFailed(
                "\(component): \(missing.count) parameters have no checkpoint tensor, e.g. "
                    + missing.sorted().prefix(5).joined(separator: ", ")
            )
        }
        guard unused.isEmpty else {
            throw H3Error.weightLoadingFailed(
                "\(component): \(unused.count) checkpoint tensors matched no parameter, e.g. "
                    + unused.sorted().prefix(5).joined(separator: ", ")
            )
        }

        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model.parameters())
        H3Debug.log("\(component): loaded \(weights.count) tensors")
    }

    // MARK: - Transformer

    /// Load the H3 transformer from `<modelDir>/transformer`. Checkpoint dtypes are kept as-is
    /// (mixed bf16 / fp32 — the fp32 modules are part of the checkpoint contract).
    /// - Parameters:
    ///   - numLayers: block count to materialize (parity harnesses use 1; nil = all).
    ///   - quantization: on-the-fly quantization applied after the weights land (MiniMax's
    ///     exclusion list keeps the fp32 modules, refiner and output norm at full precision).
    /// - Parameter turboLoRA: a step-distillation LoRA to fold into the weights before
    ///   quantization (see `H3TurboLoRA`). Folding needs the bf16 stage, so passing one disables
    ///   the prequantized fast path rather than quietly loading un-adapted weights.
    /// - Parameter component: which transformer partition to load. `ref2va` runs against
    ///   `transformer_ref`, whose `config.json` is byte-identical to the main one — the directory
    ///   name is the only difference, so the loader, the quantization filter and the prequantized
    ///   export all apply unchanged.
    public static func loadTransformer(
        modelDirectory: URL,
        numLayers: Int? = nil,
        quantization: H3Quantization = .none,
        skipPrequantizedPickup: Bool = false,
        turboLoRA: URL? = nil,
        turboLoRAStrength: Float = 1.0,
        component: String = "transformer"
    ) throws -> H3Transformer {
        let directory = modelDirectory.appendingPathComponent(component)
        var config = try H3TransformerConfig.load(from: directory.appendingPathComponent("config.json"))
        if let numLayers { config.numLayers = numLayers }
        let model = H3Transformer(config: config)

        // Prequantized fast path: quantize the (lazy) structure with the same filter, then load
        // the exported parameters — reads ~18.5 GB instead of 62 at qint8.
        if !skipPrequantizedPickup, numLayers == nil, turboLoRA == nil,
           H3PrequantizedCheckpoint.exists(
               modelDirectory: modelDirectory, component: component, quantization: quantization) {
            H3QuantizationFilter.apply(
                quantization, to: model, exclusions: H3QuantizationFilter.transformerExclusions)
            try H3PrequantizedCheckpoint.load(
                into: model,
                from: H3PrequantizedCheckpoint.url(
                    modelDirectory: modelDirectory, component: component, quantization: quantization),
                component: component,
                quantization: quantization
            )
            return model
        }

        // diffusers module names -> our @ModuleInfo keys. The LoRA ships against the same
        // diffusers tree, so its targets go through the identical rewrite.
        func transformerRemap(_ key: String) -> String {
            key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
                .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj.")
                .replacingOccurrences(of: ".ff.net.2.", with: ".ff.out.")
        }
        var weights = try loadShardedWeights(directory: directory) { key in
            if let numLayers, key.hasPrefix("transformer_blocks.") {
                let layerIndex = Int(key.split(separator: ".")[1]) ?? .max
                guard layerIndex < numLayers else { return nil }
            }
            return transformerRemap(key)
        }
        if let turboLoRA {
            try H3TurboLoRA.fold(
                into: &weights, from: turboLoRA, strength: turboLoRAStrength, remap: transformerRemap)
        }
        try apply(weights: weights, to: model, component: component)
        if quantization != .none {
            H3QuantizationFilter.apply(
                quantization, to: model, exclusions: H3QuantizationFilter.transformerExclusions)
            eval(model.parameters())
            Memory.clearCache()
            H3Debug.log("transformer quantized: \(quantization.displayName)")
        }
        return model
    }

    // MARK: - Text encoder

    /// Load the truncated Qwen3-VL conditioner from `<modelDir>/text_encoder`: layers 0..<50,
    /// embeddings only. Vision tower, layers 50..<64, final norm and LM head are skipped.
    public static func loadTextEncoder(
        modelDirectory: URL,
        numLayers: Int = H3Constants.textEncoderLayer,
        quantization: H3Quantization = .none,
        skipPrequantizedPickup: Bool = false
    ) throws -> Qwen3VLTextEncoder {
        let directory = modelDirectory.appendingPathComponent("text_encoder")
        let config = try Qwen3VLTextConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = Qwen3VLTextEncoder(config: config, numLayers: numLayers)

        if !skipPrequantizedPickup, numLayers == H3Constants.textEncoderLayer,
           H3PrequantizedCheckpoint.exists(
               modelDirectory: modelDirectory, component: "text_encoder", quantization: quantization) {
            H3QuantizationFilter.apply(quantization, to: model, exclusions: [])
            try H3PrequantizedCheckpoint.load(
                into: model,
                from: H3PrequantizedCheckpoint.url(
                    modelDirectory: modelDirectory, component: "text_encoder", quantization: quantization),
                component: "text_encoder",
                quantization: quantization
            )
            return model
        }

        let prefix = "model.language_model."
        let weights = try loadShardedWeights(directory: directory) { key in
            guard key.hasPrefix(prefix) else { return nil }  // drops lm_head + model.visual.*
            let stripped = String(key.dropFirst(prefix.count))
            if stripped == "norm.weight" { return nil }  // final norm: H3 reads pre-norm states
            if stripped.hasPrefix("layers.") {
                let layerIndex = Int(stripped.split(separator: ".")[1]) ?? .max
                guard layerIndex < numLayers else { return nil }
            }
            return stripped
        }
        try apply(weights: weights, to: model, component: "text_encoder")
        if quantization != .none {
            // Only Linear layers are quantized — embed_tokens (Embedding) stays full precision,
            // matching MiniMax's text-encoder exclusions.
            H3QuantizationFilter.apply(quantization, to: model, exclusions: [])
            eval(model.parameters())
            Memory.clearCache()
            H3Debug.log("text_encoder quantized: \(quantization.displayName)")
        }
        return model
    }
}


extension H3WeightLoader {
    /// Load the Qwen3-VL vision tower (fl2va keyframes) from `<modelDir>/text_encoder`.
    /// The full-patch conv3d patch embedding is reshaped into its equivalent Linear.
    /// - Parameter dtype: **bf16 by default, to match the released pipeline**, which loads the
    ///   whole conditioner in bf16. Running the tower in fp32 is more accurate *in isolation*
    ///   (27 pre-LN blocks compound bf16 rounding into ~5 % outliers on the merged embeds) but
    ///   it feeds the text stack a different trajectory: measured end to end, the fp32 tower
    ///   drives one image token into a massive activation at decoder layer 43 (18 176 where the
    ///   reference carries 161) and the conditioning ends at cosine 0.67 from the reference,
    ///   while the bf16 tower tracks it at cosine 0.9997. Fidelity to the release wins over
    ///   component-wise accuracy: the diffusion transformer was trained on bf16 conditioning.
    ///   Pass `.float32` to reproduce the isolated-accuracy measurement (`parity vision-tower`,
    ///   whose reference dump is fp32).
    public static func loadVisionTower(
        modelDirectory: URL, dtype: DType = .bfloat16
    ) throws -> Qwen3VLVisionTower {
        let directory = modelDirectory.appendingPathComponent("text_encoder")
        let config = try Qwen3VLVisionConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = Qwen3VLVisionTower(config: config)

        let prefix = "model.visual."
        var weights = try loadShardedWeights(directory: directory) { key in
            guard key.hasPrefix(prefix) else { return nil }
            return String(key.dropFirst(prefix.count))
                .replacingOccurrences(of: "patch_embed.proj.", with: "patch_embed.")
        }
        if let pe = weights["patch_embed.weight"] {
            weights["patch_embed.weight"] = pe.reshaped(pe.dim(0), -1)
        }
        for (key, value) in weights { weights[key] = value.asType(dtype) }
        try apply(weights: weights, to: model, component: "vision_tower")
        return model
    }
}
