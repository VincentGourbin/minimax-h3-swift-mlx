// SmokeCommand.swift - Load components against real weights and run tiny forwards
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX

struct SmokeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "smoke",
        abstract: "Load one or all components from the real checkpoint and run a tiny forward pass."
    )

    @Argument(help: "Component: audio-vae | video-vae | text-encoder | transformer | all")
    var component: String = "all"

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    @Option(name: .long, help: "Quantization probe: none | qint8 | qint6 | int4 | mxfp8 | mxfp4 | nvfp4.")
    var quant: String = "none"

    @Flag(name: .long, help: "Load from the prequantized export instead of quantizing on the fly.")
    var fromPrequantized = false

    func run() async throws {
        H3Debug.isEnabled = true
        let directory = URL(fileURLWithPath: modelsDir)
        let all = component == "all"

        if all || component == "audio-vae" {
            try smokeAudioVAE(directory)
        }
        if all || component == "video-vae" {
            try smokeVideoVAE(directory)
        }
        if all || component == "text-encoder" {
            try await smokeTextEncoder(directory)
        }
        if all || component == "transformer" {
            try smokeTransformer(directory)
        }
        print("Smoke test(s) passed.")
    }

    private func smokeAudioVAE(_ directory: URL) throws {
        print("→ audio VAE: loading 0.6 GB…")
        let start = Date()
        let vae = try H3WeightLoader.loadAudioVAE(modelDirectory: directory)
        print(String(format: "  loaded in %.1f s", Date().timeIntervalSince(start)))
        // 40 latents = 1 s of audio per channel.
        let latents = MLXRandom.normal([2, 32, 40], type: Float.self)
        let waveform = try vae.decode(latents)
        eval(waveform)
        let expected = 40 * 800
        guard waveform.shape == [2, expected] else {
            throw H3Error.generationFailed("audio decode shape \(waveform.shape), expected [2, \(expected)]")
        }
        let peak = abs(waveform).max().item(Float.self)
        print("  decode OK: (2, \(expected)) samples, peak \(peak)")
        Memory.clearCache()
    }

    private func smokeVideoVAE(_ directory: URL) throws {
        print("→ video VAE: loading 10.4 GB…")
        let start = Date()
        let vae = try H3WeightLoader.loadVideoVAE(modelDirectory: directory)
        print(String(format: "  loaded in %.1f s", Date().timeIntervalSince(start)))
        // 7 latent frames (17*1+5=22 pixel frames), 128x128 px -> 8x8 latents.
        let latents = MLXRandom.normal([1, 24, 7, 8, 8], type: Float.self)
        let video = vae.decode(latents)
        eval(video)
        guard video.shape == [1, 3, 22, 128, 128] else {
            throw H3Error.generationFailed("video decode shape \(video.shape), expected [1, 3, 22, 128, 128]")
        }
        print("  decode OK: \(video.shape)")
        Memory.clearCache()
    }

    private func smokeTextEncoder(_ directory: URL) async throws {
        print("→ text encoder: loading ~52 GB (layers 0-49)…")
        let start = Date()
        let encoder = try H3WeightLoader.loadTextEncoder(
            modelDirectory: directory,
            quantization: H3Quantization(rawValue: quant) ?? .none,
            skipPrequantizedPickup: !fromPrequantized)
        print(String(format: "  loaded in %.1f s", Date().timeIntervalSince(start)))
        let ids = MLXArray([Int32](arrayLiteral: 3838, 374, 264, 1273)).expandedDimensions(axis: 0)
        let hidden = encoder(ids)
        eval(hidden)
        guard hidden.shape == [1, 4, 5120] else {
            throw H3Error.generationFailed("text encoder shape \(hidden.shape), expected [1, 4, 5120]")
        }
        let norm = sqrt(mean(hidden.asType(.float32) * hidden.asType(.float32))).item(Float.self)
        print("  forward OK: \(hidden.shape), RMS \(norm)")
        Memory.clearCache()
    }

    private func smokeTransformer(_ directory: URL) throws {
        print("→ transformer: loading 61.7 GB…")
        let start = Date()
        let transformer = try H3WeightLoader.loadTransformer(modelDirectory: directory)
        print(String(format: "  loaded in %.1f s", Date().timeIntervalSince(start)))

        // Tiny packed sequence: 4 text tokens, 2 latent frames of 4x4 latents, 10 audio latents.
        let layout = try H3Packing.buildPackedSequence(
            textTokenTags: [1, 1, 1, 1],
            numLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            numAudioLatents: 10,
            patchSize: (1, 2, 2)
        )
        let videoRows = MLXRandom.normal([2 * 4, 96], type: Float.self)
        let audioRows = MLXRandom.normal([20, 32], type: Float.self)
        let textEmbeds = MLXRandom.normal([1, 4, 5120], type: Float.self) * 0.02
        let (timesteps, indices) = H3Packing.buildRowTimesteps(
            layout: layout, videoTimestep: 0.0, audioTimestep: 0.0,
            conditionVideoTimestep: 0.999, conditionAudioTimestep: 1.0
        )
        let (video, audio) = transformer(
            videoRows: videoRows, audioRows: audioRows,
            textEmbeds: textEmbeds.asType(.bfloat16),
            timesteps: timesteps, timestepIndices: indices, layout: layout
        )
        eval(video, audio)
        guard video.shape == [8, 96], audio.shape == [20, 32] else {
            throw H3Error.generationFailed("transformer shapes \(video.shape) / \(audio.shape)")
        }
        print("  forward OK: video \(video.shape), audio \(audio.shape)")
        Memory.clearCache()
    }
}
