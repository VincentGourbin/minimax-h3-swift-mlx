// ExportQuantizedCommand.swift - One-time quantized weight export
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX

struct ExportQuantizedCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-quantized",
        abstract: "Quantize a component once and save it under <models>/mlx-prequantized/<quant>/ "
            + "so later loads read the small file directly."
    )

    @Argument(help: "Component: transformer | text-encoder | all")
    var component: String = "all"

    @Option(name: .long, help: "Quantization: qint8 | qint6 | int4.")
    var quant: String = "qint8"

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    func run() async throws {
        H3Debug.isEnabled = true
        guard let quantization = H3Quantization(rawValue: quant), quantization != .none else {
            throw ValidationError("`--quant` must be one of: qint8, qint6, int4")
        }
        let directory = URL(fileURLWithPath: modelsDir)
        let all = component == "all"

        if all || component == "text-encoder" {
            print("Quantizing text encoder (\(quantization.rawValue))…")
            let start = Date()
            let model = try H3WeightLoader.loadTextEncoder(
                modelDirectory: directory, quantization: quantization, skipPrequantizedPickup: true)
            try H3WeightLoader.exportQuantized(
                model: model, modelDirectory: directory,
                component: "text_encoder", quantization: quantization)
            print(String(format: "  done in %.1f s", Date().timeIntervalSince(start)))
            Memory.clearCache()
        }
        if all || component == "transformer" {
            print("Quantizing transformer (\(quantization.rawValue))…")
            let start = Date()
            let model = try H3WeightLoader.loadTransformer(
                modelDirectory: directory, quantization: quantization, skipPrequantizedPickup: true)
            try H3WeightLoader.exportQuantized(
                model: model, modelDirectory: directory,
                component: "transformer", quantization: quantization)
            print(String(format: "  done in %.1f s", Date().timeIntervalSince(start)))
            Memory.clearCache()
        }
        print("Export complete.")
    }
}
