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
import MLXNN
import Tokenizers

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

    @Option(name: .long, help: "Quantization for dit-block0 (degradation probe): none | qint8 | qint6 | int4.")
    var quant: String = "none"

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

        case "video-vae-encode":
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("video_vae_encode.safetensors"))
            let vae = try H3WeightLoader.loadVideoVAE(modelDirectory: directory, includeEncoder: true)
            let moments = try vae.encodeClip(reference["pixels"]!)
            try compare(ours: moments, reference: reference["moments"]!, name: "moments")

        case "vision-tower":
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("vision_tower.safetensors"))
            let tower = try H3WeightLoader.loadVisionTower(modelDirectory: directory)
            let grid = reference["grid_thw"]!.asType(.int32)
            let (embeds, deepstack) = tower(
                reference["pixel_values"]!,
                gridH: Int(grid[0, 1].item(Int32.self)),
                gridW: Int(grid[0, 2].item(Int32.self)))
            try compare(ours: embeds, reference: reference["embeds"]!, name: "image embeds")
            try compare(ours: deepstack[0], reference: reference["deepstack_0"]!, name: "deepstack[0]")
            try compare(ours: deepstack[2], reference: reference["deepstack_2"]!, name: "deepstack[2]")

        case "keyframe-preprocess":
            // Canvas preparation must be BIT-EXACT vs PIL (LANCZOS fixed point), and the
            // patchify exactly reproduces the Qwen2-VL fast processor's block-major layout.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("keyframe_preprocess.safetensors"))
            let original = keyframeImage(from: reference["original"]!)
            let stretched = original.prepared(canvasWidth: 448, canvasHeight: 256, stretch: true)
            let covered = original.prepared(canvasWidth: 448, canvasHeight: 256, stretch: false)
            try compare(
                ours: MLXArray(stretched.pixels, [256, 448, 3]).asType(.float32),
                reference: reference["canvas_stretch"]!.asType(.float32),
                name: "canvas stretch (uint8)")
            try compare(
                ours: MLXArray(covered.pixels, [256, 448, 3]).asType(.float32),
                reference: reference["canvas_cover"]!.asType(.float32),
                name: "canvas cover (uint8)")
            let (patches, gridH, gridW) = stretched.visionPatches()
            let grid = reference["grid_thw"]!.asType(.int32)
            guard gridH == Int(grid[0, 1].item(Int32.self)),
                  gridW == Int(grid[0, 2].item(Int32.self))
            else {
                throw H3Error.generationFailed("grid \(gridH)x\(gridW) vs reference \(grid)")
            }
            try compare(ours: patches, reference: reference["pixel_values"]!, name: "pixel_values")

        case "text-layer0-mm":
            // The fl2va conditioner path through real layer 0: presentation token ids, mrope 3D
            // positions, vision-embed injection, deepstack tap 0.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("text_layer0_mm.safetensors"))
            let image = keyframeImage(from: reference["image"]!)
            let (patches, gridH, gridW) = image.visionPatches()
            try compare(ours: patches, reference: reference["pixel_values"]!, name: "pixel_values")

            let tokenizer = try await AutoTokenizer.from(
                modelFolder: directory.appendingPathComponent("tokenizer"))
            let presentation = try H3Presentation(
                prompt: "a quiet forest at dawn",
                imageGrids: [(gridH, gridW)],
                tokenizer: tokenizer)
            try compare(
                ours: MLXArray(presentation.tokenIds).asType(.float32),
                reference: reference["token_ids"]![0].asType(.float32),
                name: "presentation token ids")
            try compare(
                ours: MLXArray(presentation.mmTokenTypes).asType(.float32),
                reference: reference["mm_token_types"]![0].asType(.float32),
                name: "mm token types")
            let layout = try presentation.multimodalLayout()!
            try compare(
                ours: layout.positions.asType(.float32),
                reference: reference["position_ids"]!.asType(.float32),
                name: "mrope positions")

            let tower = try H3WeightLoader.loadVisionTower(modelDirectory: directory)
            let (imageEmbeds, deepstack) = tower(patches, gridH: gridH, gridW: gridW)
            let encoder = try H3WeightLoader.loadTextEncoder(modelDirectory: directory, numLayers: 1)
            // The reference dump runs fp32; cast the single layer to fp32 too so the check
            // isolates implementation errors from bf16 rounding (production stays bf16).
            let fp32 = Dictionary(
                uniqueKeysWithValues: encoder.parameters().flattened().map {
                    ($0.0, $0.1.asType(.float32))
                })
            encoder.update(parameters: ModuleParameters.unflattened(fp32))
            let hidden = encoder(
                MLXArray(presentation.tokenIds).expandedDimensions(axis: 0),
                imageEmbeds: imageEmbeds,
                deepstack: deepstack,
                layout: layout)
            try compare(ours: hidden, reference: reference["layer0_out"]!, name: "layer0 hidden (mm)")

        case "text-layer0":
            // Embeddings + decoder layer 0 alone: catches RoPE/GQA/norm mistakes cheaply, and
            // validates that text-only interleaved-mrope really collapses to standard RoPE.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("text_layer0.safetensors"))
            let encoder = try H3WeightLoader.loadTextEncoder(
                modelDirectory: directory, numLayers: 1)
            let hidden = encoder(reference["token_ids"]!.asType(.int32))
            try compare(ours: hidden, reference: reference["layer0_out"]!, name: "layer0 hidden")

        case "dit-block0":
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("dit_block0.safetensors"))
            let transformer = try H3WeightLoader.loadTransformer(
                modelDirectory: directory, numLayers: 1,
                quantization: H3Quantization(rawValue: quant) ?? .none)
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

    /// (H, W, 3) uint8 reference tensor -> bitmap.
    private func keyframeImage(from array: MLXArray) -> H3KeyframeImage {
        H3KeyframeImage(
            width: array.dim(1), height: array.dim(0),
            pixels: array.asType(.uint8).asArray(UInt8.self))
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
