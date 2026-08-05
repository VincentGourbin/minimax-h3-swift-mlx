// ParityCommand.swift - Numeric comparison against diffusers reference dumps
// Copyright 2026 Vincent Gourbin
//
// Counterpart of scripts/parity_reference.py: the Python side runs the reference implementation
// on fixed inputs and writes <models>/parity/*.safetensors; this command reruns the same inputs
// through the Swift port and reports max/mean absolute error.

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX

struct ParityCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "parity",
        abstract: "Compare Swift outputs against reference dumps produced by scripts/parity_reference.py."
    )

    @Argument(help: "Component: audio-vae | video-vae")
    var component: String

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    @Option(name: .long, help: "Max tolerated max-abs-error.")
    var tolerance: Float = 5e-3

    func run() async throws {
        H3Debug.isEnabled = true
        let directory = URL(fileURLWithPath: modelsDir)
        let parityDir = directory.appendingPathComponent("parity")

        switch component {
        case "audio-vae":
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("audio_vae_decode.safetensors"))
            let vae = try H3WeightLoader.loadAudioVAE(modelDirectory: directory)
            let ours = vae.decode(reference["latents"]!)
            try compare(ours: ours, reference: reference["waveform"]!.squeezed(axis: 1), name: "waveform")

        case "video-vae":
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("video_vae_decode.safetensors"))
            let vae = try H3WeightLoader.loadVideoVAE(modelDirectory: directory)
            let ours = vae.decode(reference["latents"]!)
            try compare(ours: ours, reference: reference["video"]!, name: "video")

        case "dit-block0":
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("dit_block0.safetensors"))
            let transformer = try H3WeightLoader.loadTransformer(modelDirectory: directory, numLayers: 1)
            let layout = try H3Packing.buildPackedSequence(
                textTokenTags: [1, 1, 1, 1],
                numLatentFrames: 2, latentHeight: 4, latentWidth: 4,
                numAudioLatents: 10, patchSize: (1, 2, 2)
            )
            // Bonus check: the packing grid itself, against the reference fp64 grid.
            try compare(
                ours: layout.positionIds,
                reference: reference["position_ids"]!.asType(.float32),
                name: "position_ids")
            let (timesteps, timestepIndices) = H3Packing.buildRowTimesteps(
                layout: layout, videoTimestep: 0.0, audioTimestep: 0.0,
                conditionVideoTimestep: 0.999, conditionAudioTimestep: 1.0
            )
            let (video, audio) = transformer(
                videoRows: reference["video_rows"]![0],
                audioRows: reference["audio_rows"]![0],
                textEmbeds: reference["text_embeds"]!.asType(.bfloat16),
                timesteps: timesteps, timestepIndices: timestepIndices, layout: layout
            )
            try compare(ours: video, reference: reference["video_out"]![0], name: "video velocity")
            try compare(ours: audio, reference: reference["audio_out"]![0], name: "audio velocity")

        default:
            throw ValidationError("Unknown component '\(component)'.")
        }
    }

    private func compare(ours: MLXArray, reference: MLXArray, name: String) throws {
        guard ours.shape == reference.shape else {
            throw H3Error.generationFailed("\(name): shape \(ours.shape) vs reference \(reference.shape)")
        }
        let difference = abs(ours.asType(.float32) - reference.asType(.float32))
        let maxError = difference.max().item(Float.self)
        let meanError = mean(difference).item(Float.self)
        let refScale = abs(reference.asType(.float32)).max().item(Float.self)
        print("\(name): max|Δ| \(maxError)  mean|Δ| \(meanError)  (ref scale \(refScale))")
        guard maxError <= tolerance else {
            throw H3Error.generationFailed("\(name): max error \(maxError) exceeds tolerance \(tolerance)")
        }
        print("PARITY OK")
    }
}
