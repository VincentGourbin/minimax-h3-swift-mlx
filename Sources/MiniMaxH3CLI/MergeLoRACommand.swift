// MergeLoRACommand.swift - Bake a Turbo LoRA into a copy of the transformer, shard by shard
// Copyright 2026 Vincent Gourbin
//
// Folding at load time does not fit in 96 GB, and the failure is silent: the fold materializes the
// ~55 GB of attention and feed-forward weights that were previously memory-mapped (and therefore
// evictable), and quantization then wants another 18.5 GB on top. Both attempts were killed by the
// OS right after "transformer: loaded 638 tensors", with no error to read.
//
// So the fold happens OFFLINE, one shard at a time: peak stays at a single 4.8 GB shard plus its
// float32 transients, the output is an ordinary diffusers-layout transformer, and every later run
// takes the normal (memory-mapped, prequantizable) path with no idea a LoRA was ever involved.
// Run `export-quantized` on the result and the Turbo weights load in seconds like any other.

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX

struct MergeLoRACommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "merge-lora",
        abstract: "Fold a LoRA into the transformer weights offline and write a new model directory."
    )

    @Option(name: .long, help: "LoRA .safetensors to fold (e.g. a 4-step Turbo adapter).")
    var lora: String

    @Option(name: .long, help: "Output model directory; siblings are symlinked from the source.")
    var out: String

    @Option(name: .long, help: "Multiplier on the folded delta (sharpness dial; 1.0 = as published).")
    var strength: Double = 1.0

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    /// Components the pipeline reads besides the transformer; symlinked so `--models-dir <out>`
    /// is a drop-in replacement instead of a 100 GB copy.
    ///
    /// `mlx-prequantized` is deliberately NOT among them. That directory holds the quantized
    /// transformer exported from the ORIGINAL weights, and the loader prefers it over the shards —
    /// linking it here would make every run silently load un-adapted weights at the Turbo step
    /// count, which looks like a quality collapse rather than a wiring mistake.
    private static let siblings = [
        "text_encoder", "vae", "audio_vae", "tokenizer", "processor", "scheduler",
        "audio_scheduler",
    ]

    func run() async throws {
        H3Debug.isEnabled = true
        let source = URL(fileURLWithPath: modelsDir)
        let destination = URL(fileURLWithPath: out)
        let loraURL = URL(fileURLWithPath: lora)
        guard FileManager.default.fileExists(atPath: loraURL.path) else {
            throw ValidationError("no LoRA file at \(loraURL.path)")
        }

        let sourceTransformer = source.appendingPathComponent("transformer")
        let destinationTransformer = destination.appendingPathComponent("transformer")
        try FileManager.default.createDirectory(
            at: destinationTransformer, withIntermediateDirectories: true)

        // config.json is small and must travel with the weights.
        let config = sourceTransformer.appendingPathComponent("config.json")
        let configCopy = destinationTransformer.appendingPathComponent("config.json")
        try? FileManager.default.removeItem(at: configCopy)
        try FileManager.default.copyItem(at: config, to: configCopy)

        // Mirror each sibling as a REAL directory of file symlinks, never as a symlinked
        // directory: `FileManager.contentsOfDirectory` refuses a symlink-to-directory, which is how
        // the first merged checkpoint denoised all four steps and then died on "the file 'vae'
        // couldn't be opened" — after the expensive part was already done.
        for sibling in Self.siblings {
            let target = source.appendingPathComponent(sibling)
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: target, includingPropertiesForKeys: nil) else { continue }
            let mirror = destination.appendingPathComponent(sibling)
            try FileManager.default.createDirectory(at: mirror, withIntermediateDirectories: true)
            for file in files {
                let link = mirror.appendingPathComponent(file.lastPathComponent)
                try? FileManager.default.removeItem(at: link)
                try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
            }
        }

        // The prequantized directory is per-file: keep the text encoder's fast path (it is
        // untouched by a transformer LoRA) and leave `transformer.safetensors` out, so
        // `export-quantized` writes the adapted one here later.
        let sourcePrequantized = source.appendingPathComponent("mlx-prequantized")
        if let quantDirectories = try? FileManager.default.contentsOfDirectory(
            at: sourcePrequantized, includingPropertiesForKeys: nil) {
            for quantDirectory in quantDirectories {
                let mirror = destination.appendingPathComponent("mlx-prequantized")
                    .appendingPathComponent(quantDirectory.lastPathComponent)
                try FileManager.default.createDirectory(at: mirror, withIntermediateDirectories: true)
                let files = (try? FileManager.default.contentsOfDirectory(
                    at: quantDirectory, includingPropertiesForKeys: nil)) ?? []
                for file in files where file.lastPathComponent != "transformer.safetensors" {
                    let link = mirror.appendingPathComponent(file.lastPathComponent)
                    try? FileManager.default.removeItem(at: link)
                    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
                }
            }
        }

        let shards = try FileManager.default
            .contentsOfDirectory(at: sourceTransformer, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else {
            throw H3Error.modelNotFound("No .safetensors shards in \(sourceTransformer.path)")
        }

        var total = 0
        let start = Date()
        for (index, shard) in shards.enumerated() {
            var arrays = try loadArrays(url: shard)
            // Both sides carry diffusers keys here — the port's own renames happen at load time,
            // downstream of this file — so the fold needs no rewrite at all.
            let folded = try H3TurboLoRA.fold(
                into: &arrays, from: loraURL, strength: Float(strength),
                requireMatch: false, remap: { $0 })
            total += folded
            try save(arrays: arrays, url: destinationTransformer
                .appendingPathComponent(shard.lastPathComponent))
            arrays.removeAll()
            Memory.clearCache()
            print("  shard \(index + 1)/\(shards.count): \(folded) tensor(s) folded")
        }

        guard total > 0 else {
            throw H3Error.invalidConfiguration(
                "the LoRA matched no tensor in any shard — wrong adapter for this checkpoint?")
        }
        print(String(
            format: "Folded %d tensors in %.0f s. Use it with:\n"
                + "  minimax-h3 export-quantized transformer --models-dir %@\n"
                + "  minimax-h3 generate \"…\" --models-dir %@ -s 5",
            total, Date().timeIntervalSince(start), destination.path, destination.path))
    }
}
