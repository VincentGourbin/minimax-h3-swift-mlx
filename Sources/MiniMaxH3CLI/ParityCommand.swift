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
            // Isolated accuracy: the reference dump runs the tower in fp32, so this probe does
            // too. Production loads it in bf16 to match the released pipeline end to end — see
            // `loadVisionTower` and the `conditioner-fl2va` probe.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("vision_tower.safetensors"))
            let tower = try H3WeightLoader.loadVisionTower(
                modelDirectory: directory, dtype: .float32)
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

            // fp32 tower here too: this probe's reference dump builds the tower in fp32 and the
            // point is to isolate layer 0's arithmetic, not to reproduce the release's dtypes.
            let tower = try H3WeightLoader.loadVisionTower(
                modelDirectory: directory, dtype: .float32)
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

        case "conditioner-fl2va":
            // Full-depth E2E: real keyframe + prompt -> hidden_states[50] with the production
            // paths (bf16 text stack, fp32 vision tower) against the all-bf16 reference dump.
            // Judged on relative RMS: per-component parity is exact (see the other probes), so
            // what remains here is compounded bf16 rounding, not implementation error.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("conditioner_fl2va.safetensors"))
            let canvas = keyframeImage(from: reference["canvas"]!)
            let (patches, gridH, gridW) = canvas.visionPatches()
            let tokenizer = try await AutoTokenizer.from(
                modelFolder: directory.appendingPathComponent("tokenizer"))
            let presentation = try H3Presentation(
                prompt: "A red fox walking through a snowy forest at dusk, soft wind, distant birdsong.",
                imageGrids: [(gridH, gridW)],
                tokenizer: tokenizer)
            try compare(
                ours: MLXArray(presentation.tokenIds).asType(.float32),
                reference: reference["token_ids"]![0].asType(.float32),
                name: "presentation token ids")
            let layout = try presentation.multimodalLayout()!

            // Both the reference and production run the tower in bf16; `--quant fp32-tower`
            // reproduces the measurement that made fp32 look preferable in isolation, and shows
            // how it derails the deep trajectory.
            let towerDType: DType = quant == "fp32-tower" ? .float32 : .bfloat16
            let tower = try H3WeightLoader.loadVisionTower(
                modelDirectory: directory, dtype: towerDType)
            let (imageEmbeds, deepstack) = tower(patches, gridH: gridH, gridW: gridW)
            let encoder = try H3WeightLoader.loadTextEncoder(modelDirectory: directory)
            let depths = [1, 2, 3, 4, 5, 10, 20, 30, 40, 42, 44, 45, 46, 47, 48, 49]
            let (hidden, captured) = encoder.forward(
                MLXArray(presentation.tokenIds).expandedDimensions(axis: 0),
                imageEmbeds: imageEmbeds,
                deepstack: deepstack,
                layout: layout,
                captureDepths: Set(depths))

            // Depth profile: a bug jumps at one layer, bf16 rounding ramps smoothly. The last
            // two columns track the cell that dominates the final error, in both runs.
            let (probeRow, probeColumn) = (32, 731)
            print("depth   rel RMS      cosine       max|Δ|      RMS ours/ref     cell(32,731) ours/ref")
            for depth in depths {
                guard let mine = captured[depth],
                      let theirs = reference["hidden_\(depth)"] else { continue }
                let a = mine.asType(.float32)
                let b = theirs.asType(.float32)
                let d = a - b
                let rms = { (t: MLXArray) in sqrt(mean(t * t)).item(Float.self) }
                let cos = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))).item(Float.self)
                print(String(
                    format: "%5d   %-11.5g  %-11.6f  %-10.4g  %.4g / %.4g     %.5g / %.5g",
                    depth, rms(d) / rms(b), cos, abs(d).max().item(Float.self), rms(a), rms(b),
                    a[0, probeRow, probeColumn].item(Float.self),
                    b[0, probeRow, probeColumn].item(Float.self)))
            }

            // Decisive test: feed the REFERENCE's own state at depth 42 through our layers 42-43.
            // Matching here means our arithmetic is right and the depth-50 gap is accumulated
            // input drift amplified by the model; exploding here means those layers are wrong.
            if let seed = reference["hidden_42"], let target = reference["hidden_44"] {
                let replayed = encoder.applyLayers(
                    seed.asType(.bfloat16), range: 42..<44, layout: layout).asType(.float32)
                let goal = target.asType(.float32)
                let d = replayed - goal
                let rel = (sqrt(mean(d * d)) / sqrt(mean(goal * goal))).item(Float.self)
                print(String(
                    format: "replay 42->44 from the reference's own state: rel RMS %.5g  max|Δ| %.4g"
                        + "  cell(32,731) %.5g vs %.5g",
                    rel, abs(d).max().item(Float.self),
                    replayed[0, probeRow, probeColumn].item(Float.self),
                    goal[0, probeRow, probeColumn].item(Float.self)))

                // Same replay, but with that single cell nudged to the value our own run carried
                // at depth 42. If one cell's few-percent difference is enough to blow the output
                // up, the knife-edge belongs to the model, not to either implementation.
                if let mine42 = captured[42] {
                    let perturbed = seed.asType(.bfloat16)
                    perturbed[0, probeRow, probeColumn] =
                        mine42[0, probeRow, probeColumn].asType(.bfloat16)
                    let out = encoder.applyLayers(perturbed, range: 42..<44, layout: layout)
                    print(String(
                        format: "same replay, cell(32,731) nudged %.5g -> %.5g: output %.5g "
                            + "(unperturbed run gave %.5g)",
                        goal[0, probeRow, probeColumn].item(Float.self),
                        mine42[0, probeRow, probeColumn].asType(.float32).item(Float.self),
                        out.asType(.float32)[0, probeRow, probeColumn].item(Float.self),
                        replayed[0, probeRow, probeColumn].item(Float.self)))
                }
            }

            let ours = hidden.asType(.float32)
            let ref = reference["hidden_50"]!.asType(.float32)
            guard ours.shape == ref.shape else {
                throw H3Error.generationFailed("hidden: shape \(ours.shape) vs \(ref.shape)")
            }
            let difference = ours - ref
            // Where does the error live? Qwen stacks grow "massive activations" — a couple of
            // (token, channel) cells worth thousands — and a single mismatched cell can dominate
            // an RMS built over 700k elements. Print the worst cells, then the metric with them
            // excluded, so the two failure modes stay distinguishable.
            let flatDiff = abs(difference).reshaped(-1)
            let order = argSort(flatDiff)
            let width = ours.dim(-1)
            print("worst cells (token, channel): ours vs ref")
            for rank in 0..<5 {
                let flatIndex = Int(order[order.dim(0) - 1 - rank].item(Int32.self))
                let (row, column) = (flatIndex / width, flatIndex % width)
                print(String(
                    format: "  (%3d, %4d)  ours %-12.6g ref %-12.6g  Δ %.4g",
                    row, column,
                    ours[0, row, column].item(Float.self),
                    ref[0, row, column].item(Float.self),
                    flatDiff[flatIndex].item(Float.self)))
            }
            let sorted = MLX.sorted(flatDiff)
            let cutoff = sorted[sorted.dim(0) - 5].item(Float.self)
            let masked = MLX.where(flatDiff .< cutoff, flatDiff, MLXArray(Float(0)))
            print(String(
                format: "rel RMS excluding those 5 cells: %.5g",
                (sqrt(mean(masked * masked)) / sqrt(mean(ref * ref))).item(Float.self)))

            let relRMS = (sqrt(mean(difference * difference))
                / sqrt(mean(ref * ref))).item(Float.self)
            let cosine = (sum(ours * ref)
                / (sqrt(sum(ours * ours)) * sqrt(sum(ref * ref)))).item(Float.self)
            let maxAbs = abs(difference).max().item(Float.self)
            print("hidden_states[50]: rel RMS \(relRMS)  cosine \(cosine)  max|Δ| \(maxAbs)")
            guard relRMS <= tolerance else {
                throw H3Error.generationFailed(
                    "hidden_states[50]: rel RMS \(relRMS) exceeds tolerance \(tolerance)")
            }
            print("PARITY OK")

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
