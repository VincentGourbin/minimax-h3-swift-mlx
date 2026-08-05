// PrequantizedCheckpoint.swift - Export/load quantized weights (flux-2-swift-mlx pattern)
// Copyright 2026 Vincent Gourbin
//
// On-the-fly quantization still reads the full bf16 checkpoint (62 GB for the transformer)
// before shrinking it. Exporting the quantized parameters once to
// `<modelDir>/mlx-prequantized/<quant>/<component>.safetensors` cuts every later load to the
// quantized size (~18.5 GB at qint8). The loaders pick these files up automatically when the
// requested quantization matches; metadata + strict two-way key checks guard against stale or
// mismatched exports.

import Foundation
import MLX
import MLXNN

public enum H3PrequantizedCheckpoint {
    public static func url(modelDirectory: URL, component: String, quantization: H3Quantization) -> URL {
        modelDirectory
            .appendingPathComponent("mlx-prequantized")
            .appendingPathComponent(quantization.rawValue)
            .appendingPathComponent("\(component).safetensors")
    }

    /// Save a quantized model's parameters, atomically, with identifying metadata.
    static func export(
        model: Module,
        modelDirectory: URL,
        component: String,
        quantization: H3Quantization
    ) throws {
        let destination = url(
            modelDirectory: modelDirectory, component: component, quantization: quantization)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        var arrays = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() { arrays[key] = value }

        let descriptor = quantization.descriptor!
        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".tmp-\(destination.lastPathComponent)")
        try MLX.save(
            arrays: arrays,
            metadata: [
                "format": "minimax-h3-swift-mlx-prequantized-v1",
                "component": component,
                "quantization": quantization.rawValue,
                "bits": "\(descriptor.bits)",
                "group_size": "\(descriptor.groupSize)",
            ],
            url: temporary
        )
        _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)
        H3Debug.log("exported \(component) [\(quantization.rawValue)] -> \(destination.path)")
    }

    /// Load a prequantized file into a model whose structure has already been quantized with the
    /// SAME filter. Verifies metadata and both directions of the key set.
    static func load(
        into model: Module,
        from fileURL: URL,
        component: String,
        quantization: H3Quantization
    ) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: fileURL)
        guard metadata["format"] == "minimax-h3-swift-mlx-prequantized-v1",
              metadata["component"] == component,
              metadata["quantization"] == quantization.rawValue else {
            throw H3Error.weightLoadingFailed(
                "\(fileURL.lastPathComponent): metadata mismatch (\(metadata))")
        }
        try H3WeightLoader.apply(weights: arrays, to: model, component: "\(component) [prequantized]")
    }

    static func exists(modelDirectory: URL, component: String, quantization: H3Quantization) -> Bool {
        quantization != .none
            && FileManager.default.fileExists(
                atPath: url(
                    modelDirectory: modelDirectory, component: component, quantization: quantization
                ).path)
    }
}

extension H3WeightLoader {
    /// Public entry for the export command.
    public static func exportQuantized(
        model: Module,
        modelDirectory: URL,
        component: String,
        quantization: H3Quantization
    ) throws {
        try H3PrequantizedCheckpoint.export(
            model: model, modelDirectory: modelDirectory,
            component: component, quantization: quantization)
    }
}
