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

    @Argument(help: "Component: audio-vae | video-vae | video-vae-encode | vision-tower | text-layer0 | dit-block0 | keyframe-preprocess | text-layer0-mm | conditioner-fl2va | ref-normalize | audio-vae-encode | video-condition | conditioner-ref2va")
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

            // The graph-compiled block must produce the same numbers as the eager one — fusion
            // may reassociate float ops, so identical-to-eager is checked at bf16 noise level.
            transformer.compileBlocks = true
            let (videoCompiled, audioCompiled) = transformer(
                videoRows: reference["video_rows"]![0],
                audioRows: reference["audio_rows"]![0],
                textEmbeds: reference["text_embeds"]!.asType(.bfloat16),
                timesteps: timesteps, timestepIndices: timestepIndices, layout: layout
            )
            try compare(ours: videoCompiled, reference: video, name: "video velocity (compiled vs eager)")
            try compare(ours: audioCompiled, reference: audio, name: "audio velocity (compiled vs eager)")

        case "conditioner-ref2va":
            // Full-depth E2E for ref2va: an audio reference, a soundtrack-bearing video reference
            // and an image reference, normalized and presented by the port, through the bf16 text
            // stack to hidden_states[50] against the all-bf16 reference dump. Judged on relative
            // RMS and cosine, like the fl2va probe: every component underneath is exact to its own
            // probe, so what remains here is compounded bf16 rounding.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("conditioner_ref2va.safetensors"))
            let numFrames = 124

            let voice = audioBuffer(from: reference["voice"]!, sampleRate: 16_000)
            let clipAudio = audioBuffer(from: reference["clip_audio"]!, sampleRate: 32_000)
            let clip = frames(from: reference["clip"]!)
            let picture = keyframeImage(from: reference["picture"]!)
            let references: [H3Reference] = [
                .audio(H3AudioReference(audio: voice)),
                .video(H3VideoReference(frames: clip, fps: 24, audio: clipAudio)),
                .image(H3ImageReference(image: picture)),
            ]
            let normalized = try H3ReferenceNormalizer.normalize(references, numFrames: numFrames)

            // Presentation: labels numbered per modality in request order, one timestamped vision
            // block per merged frame pair.
            var presentationReferences = [H3PresentationReference]()
            var visionBlocks = [(patches: MLXArray, gridH: Int, gridW: Int)]()
            for entry in normalized {
                switch entry {
                case .audio:
                    presentationReferences.append(.audio)
                case .video(let video):
                    let prepared = try H3ReferenceVideoProcessor.prepare(frames: video.frames)
                    for block in prepared.blocks {
                        visionBlocks.append((block, prepared.gridH, prepared.gridW))
                    }
                    presentationReferences.append(.videoWithAudio(
                        gridH: prepared.gridH, gridW: prepared.gridW,
                        blockTimestamps: prepared.blockTimestamps))
                case .image(let entry):
                    let (patches, gridH, gridW) = entry.image.visionPatches()
                    visionBlocks.append((patches, gridH, gridW))
                    presentationReferences.append(.image(gridH: gridH, gridW: gridW))
                }
            }
            let tokenizer = try await AutoTokenizer.from(
                modelFolder: directory.appendingPathComponent("tokenizer"))
            let presentation = try H3Presentation.ref2va(
                prompt: "subject_definitions:\n<Subject 1> is the room in <Picture 1>.\n\n"
                    + "summary:\n[reference generation + audio reference] A short handheld take "
                    + "in <Subject 1>.",
                references: presentationReferences, tokenizer: tokenizer)
            try compare(
                ours: MLXArray(presentation.tokenIds).asType(.float32),
                reference: reference["token_ids"]![0].asType(.float32),
                name: "presentation token ids", tolerance: 0)
            try compare(
                ours: MLXArray(presentation.mmTokenTypes).asType(.float32),
                reference: reference["mm_token_type_ids"]![0].asType(.float32),
                name: "mm token types", tolerance: 0)
            try compare(
                ours: MLXArray(presentation.tokenTags).asType(.float32),
                reference: reference["token_tags"]![0].asType(.float32),
                name: "H3 row tags", tolerance: 0)
            let layout = try presentation.multimodalLayout()!

            // `--quant reference-vision` drives the stack from the REFERENCE's own vision side —
            // its bf16 features and its bf16 deepstack taps — leaving nothing of our tower in the
            // input. This is the fl2va "replay from the reference's own state" test: if depth 50
            // then agrees, our text stack is exact and the whole gap is the tower's bf16 rounding;
            // if it still diverges, the stack itself is wrong at this sequence length.
            let imageEmbeds: MLXArray
            let deepstack: [MLXArray]
            if quant == "reference-vision" {
                let vision = try loadArrays(
                    url: parityDir.appendingPathComponent("ref2va_embeddings.safetensors"))
                guard vision["token_ids"]!.shape == reference["token_ids"]!.shape,
                      abs(vision["token_ids"]!.asType(.float32)
                          - reference["token_ids"]!.asType(.float32)).max().item(Float.self) == 0
                else {
                    throw H3Error.generationFailed(
                        "the two ref2va dumps describe different requests; regenerate both.")
                }
                // Presentation order is [video block 1, video block 2, image]; the dump batches
                // per modality, so the video features come first and the image last.
                imageEmbeds = concatenated(
                    [vision["video_features"]!, vision["image_features"]!], axis: 0)
                deepstack = (0..<3).map { level in
                    concatenated(
                        [vision["video_deepstack_\(level)"]!, vision["image_deepstack_\(level)"]!],
                        axis: 0)
                }
                print("driving the stack from the reference's own bf16 vision features")
            } else {
                // `--quant fp32-tower` asks the opposite question of `reference-vision`: our own
                // tower, but at full precision. Our fp32 output sits ~9e-6 from the reference's
                // fp32, i.e. closer to the fp32 truth than the reference's OWN bf16 is (0.019) —
                // so on distance alone fp32 should condition closer to the release than our bf16
                // does. fl2va measured the opposite, which is why this is run and not reasoned.
                let towerDType: DType = quant == "fp32-tower" ? .float32 : .bfloat16
                let tower = try H3WeightLoader.loadVisionTower(
                    modelDirectory: directory, dtype: towerDType)
                var embedsPerBlock = [MLXArray]()
                var deepstackPerBlock = [[MLXArray]]()
                for block in visionBlocks {
                    let (embeds, taps) = tower(block.patches, gridH: block.gridH, gridW: block.gridW)
                    embedsPerBlock.append(embeds)
                    deepstackPerBlock.append(taps)
                    eval([embeds] + taps)
                }
                imageEmbeds = concatenated(embedsPerBlock, axis: 0)
                deepstack = (0..<deepstackPerBlock[0].count).map { level in
                    concatenated(deepstackPerBlock.map { $0[level] }, axis: 0)
                }
            }
            eval([imageEmbeds] + deepstack)
            Memory.clearCache()

            let encoder = try H3WeightLoader.loadTextEncoder(modelDirectory: directory)
            let depths = [1, 2, 5, 10, 20, 30, 40, 45, 49]
            let (hidden, captured) = encoder.forward(
                MLXArray(presentation.tokenIds).expandedDimensions(axis: 0),
                imageEmbeds: imageEmbeds,
                deepstack: deepstack,
                layout: layout,
                captureDepths: Set(depths))

            print("depth   rel RMS      cosine       max|Δ|      RMS ours/ref")
            for depth in depths + [50] {
                let mine = depth == 50 ? hidden : captured[depth]
                guard let mine, let theirs = reference["hidden_\(depth)"] else { continue }
                let a = mine.asType(.float32)
                let b = theirs.asType(.float32)
                let d = a - b
                let rms = { (t: MLXArray) in sqrt(mean(t * t)).item(Float.self) }
                let cos = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))).item(Float.self)
                print(String(
                    format: "%5d   %-11.5g  %-11.6f  %-10.4g  %.4g / %.4g",
                    depth, rms(d) / rms(b), cos, abs(d).max().item(Float.self), rms(a), rms(b)))
            }
            let final = hidden.asType(.float32)
            let goal = reference["hidden_50"]!.asType(.float32)
            let delta = final - goal
            let relative = (sqrt(mean(delta * delta)) / sqrt(mean(goal * goal))).item(Float.self)
            let cosine = (sum(final * goal)
                / (sqrt(sum(final * final)) * sqrt(sum(goal * goal)))).item(Float.self)
            print(String(format: "hidden_50: rel RMS %.5g  cosine %.6f", relative, cosine))
            guard relative <= tolerance else {
                throw H3Error.generationFailed(
                    "hidden_50 rel RMS \(relative) exceeds tolerance \(tolerance)")
            }
            print("PARITY OK")

        case "ref2va-embeddings":
            // The discriminator: `hidden_states[0]` — token embeddings with the vision features
            // injected and NOT ONE decoder layer run. If this matches, every difference the
            // full-depth probe reports belongs to the text stack; if it does not, the conditioning
            // going in is already wrong and the depth profile is downstream of it. The raw
            // per-modality features come along so a mismatch lands on the tower or on the
            // injection rather than "somewhere in between".
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("ref2va_embeddings.safetensors"))
            let (presentation, visionBlocks) = try await ref2vaSyntheticPresentation(
                from: reference, modelDirectory: directory)
            try compare(
                ours: MLXArray(presentation.tokenIds).asType(.float32),
                reference: reference["token_ids"]![0].asType(.float32),
                name: "presentation token ids", tolerance: 0)

            let tower = try H3WeightLoader.loadVisionTower(
                modelDirectory: directory, dtype: .bfloat16)
            var embedsPerBlock = [MLXArray]()
            var deepstackPerBlock = [[MLXArray]]()
            for block in visionBlocks {
                let (embeds, taps) = tower(block.patches, gridH: block.gridH, gridW: block.gridW)
                embedsPerBlock.append(embeds)
                deepstackPerBlock.append(taps)
                eval([embeds] + taps)
            }
            // The dump batches per modality — the videos together, then the image — while the
            // presentation runs in request order, so each is compared against its own batch.
            let videoBlockCount = embedsPerBlock.count - 1
            let ourVideo = concatenated(Array(embedsPerBlock.prefix(videoBlockCount)), axis: 0)
            let ourImage = embedsPerBlock[videoBlockCount]
            try compare(
                ours: ourVideo, reference: reference["video_features"]!,
                name: "video features: ours bf16 vs theirs bf16")
            try compare(
                ours: ourImage, reference: reference["image_features"]!,
                name: "image features: ours bf16 vs theirs bf16")

            // The four corners. Two bf16 implementations of a 27-layer tower always diverge; the
            // question is whether OURS lands further from the fp32 truth than THEIRS does. If the
            // two distances match, the divergence is rounding and not a defect.
            if let imageFP32 = reference["image_features_fp32"],
               let videoFP32 = reference["video_features_fp32"] {
                try compare(
                    ours: ourImage, reference: imageFP32,
                    name: "image features: ours bf16 vs theirs FP32")
                try compare(
                    ours: reference["image_features"]!, reference: imageFP32,
                    name: "image features: theirs bf16 vs theirs FP32")
                try compare(
                    ours: ourVideo, reference: videoFP32,
                    name: "video features: ours bf16 vs theirs FP32")
                try compare(
                    ours: reference["video_features"]!, reference: videoFP32,
                    name: "video features: theirs bf16 vs theirs FP32")

                // And our own fp32 tower against their fp32, to confirm the arithmetic itself.
                let fp32Tower = try H3WeightLoader.loadVisionTower(
                    modelDirectory: directory, dtype: .float32)
                var fp32Blocks = [MLXArray]()
                for block in visionBlocks {
                    fp32Blocks.append(
                        fp32Tower(block.patches, gridH: block.gridH, gridW: block.gridW).embeds)
                }
                eval(fp32Blocks)
                try compare(
                    ours: fp32Blocks[videoBlockCount], reference: imageFP32,
                    name: "image features: ours FP32 vs theirs FP32")
            }

            let imageEmbeds = concatenated(embedsPerBlock, axis: 0)
            let deepstack = (0..<deepstackPerBlock[0].count).map { level in
                concatenated(deepstackPerBlock.map { $0[level] }, axis: 0)
            }
            eval([imageEmbeds] + deepstack)
            Memory.clearCache()
            let layout = try presentation.multimodalLayout()!
            let encoder = try H3WeightLoader.loadTextEncoder(modelDirectory: directory, numLayers: 1)
            let (_, captured) = encoder.forward(
                MLXArray(presentation.tokenIds).expandedDimensions(axis: 0),
                imageEmbeds: imageEmbeds, deepstack: deepstack, layout: layout,
                captureDepths: [0])
            try compare(
                ours: captured[0]!, reference: reference["hidden_0"]!,
                name: "hidden_0 (embeddings + vision injection, no layers)")

        case "vision-tower-large":
            // The tower at a ref2va image reference's geometry: a 128x128 grid, 16 384 tokens in
            // one call, against the few hundred every earlier tower probe used. fp32 both sides.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("vision_tower_large.safetensors"))
            let picture = keyframeImage(from: reference["image"]!)
            let (patches, gridH, gridW) = picture.visionPatches()
            let grid = reference["grid_thw"]!.asType(.int32)
            guard gridH == Int(grid[0, 1].item(Int32.self)),
                  gridW == Int(grid[0, 2].item(Int32.self))
            else {
                throw H3Error.generationFailed("grid \(gridH)x\(gridW) vs reference \(grid)")
            }
            let tower = try H3WeightLoader.loadVisionTower(
                modelDirectory: directory, dtype: .float32)
            let (embeds, deepstack) = tower(patches, gridH: gridH, gridW: gridW)
            try compare(ours: embeds, reference: reference["embeds"]!, name: "image embeds (128x128)")
            try compare(
                ours: deepstack[0], reference: reference["deepstack_0"]!, name: "deepstack[0]")
            try compare(
                ours: deepstack[2], reference: reference["deepstack_2"]!, name: "deepstack[2]")

        case "video-condition":
            // ref2va Phase B: Qwen3-VL's *video* processor on a reference clip, and the claim the
            // port is built on — a `grid_t = N` tower call equals N independent `grid_t = 1` calls,
            // because Qwen3-VL segments its vision attention per temporal group and its rotary
            // table has no temporal axis. The dump carries both so the equality is checked here
            // rather than assumed.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("video_condition.safetensors"))
            let clip = frames(from: reference["frames"]!)
            let prepared = try H3ReferenceVideoProcessor.prepare(frames: clip)
            let grid = reference["grid_thw"]!.asType(.int32)
            let (gridT, gridH, gridW) = (
                Int(grid[0, 0].item(Int32.self)), Int(grid[0, 1].item(Int32.self)),
                Int(grid[0, 2].item(Int32.self)))
            guard prepared.blocks.count == gridT, prepared.gridH == gridH, prepared.gridW == gridW
            else {
                throw H3Error.generationFailed(
                    "video grid \(prepared.blocks.count)x\(prepared.gridH)x\(prepared.gridW) "
                        + "vs reference \(gridT)x\(gridH)x\(gridW)")
            }
            let expectedLabels = reference["timestamps"]!.asType(.float32).asArray(Float.self)
                .map { H3ReferenceVideoProcessor.timestampLabel(Double($0)) }
            let ourLabels = prepared.blockTimestamps.map(H3ReferenceVideoProcessor.timestampLabel)
            guard ourLabels == expectedLabels else {
                throw H3Error.generationFailed("block labels \(ourLabels) vs \(expectedLabels)")
            }
            print("block labels: \(ourLabels.joined(separator: " ")) — OK")

            try compare(
                ours: concatenated(prepared.blocks, axis: 0),
                reference: reference["pixel_values"]!, name: "pixel_values_videos")

            let tower = try H3WeightLoader.loadVisionTower(
                modelDirectory: directory, dtype: .float32)
            var embedsPerBlock = [MLXArray]()
            for block in prepared.blocks {
                embedsPerBlock.append(tower(block, gridH: gridH, gridW: gridW).embeds)
            }
            // Against the per-group reference AND against the whole-clip one: the two are the same
            // tensor on the Python side, and that is the point.
            try compare(
                ours: concatenated(embedsPerBlock, axis: 0),
                reference: reference["embeds_per_group"]!, name: "vision embeds (per group)")
            try compare(
                ours: concatenated(embedsPerBlock, axis: 0),
                reference: reference["embeds"]!, name: "vision embeds (vs whole-clip call)")

        case "audio-vae-encode":
            // ref2va Phase B: the DAC trunk, the causal-attention `pre_block` and `mean_proj`.
            // The reference runs fp32 and so does the port, so this is an implementation check,
            // not a dtype one — expect ~1e-5, the decoder's level.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("audio_vae_encode.safetensors"))
            let vae = try H3WeightLoader.loadAudioVAE(
                modelDirectory: directory, includeEncoder: true, includeDecoder: false)
            let (mean, logStd) = try vae.encode(reference["waveform"]!)
            try compare(ours: mean, reference: reference["mean"]!, name: "posterior mean")
            try compare(ours: logStd, reference: reference["logs"]!, name: "posterior log std")

        case "ref-normalize":
            // ref2va Phase A. Two of the four contracts must be BIT-EXACT: a video already at
            // 24 fps and already on its canvas passes through untouched, and the frame-rate
            // resample picks whole frames. The two rescale paths (LANCZOS) and the soundtrack
            // resample (torchaudio's sinc kernel) carry their own tolerances.
            let reference = try loadArrays(
                url: parityDir.appendingPathComponent("ref_normalize.safetensors"))
            let numFrames = 124

            // 1. Pass-through: 24 fps, already 1344x768.
            let passthroughSource = frames(from: reference["passthrough_source"]!)
            let passthrough = try H3ReferenceNormalizer.normalizeVideoFrames(
                passthroughSource, fps: 24, numFrames: numFrames)
            try compare(
                ours: frameStack(passthrough),
                reference: reference["passthrough"]!.asType(.float32),
                name: "video pass-through (bit-exact)", tolerance: 0)

            // 2. 30 fps -> 24 fps + LANCZOS onto the video's own canvas.
            let videoSource = frames(from: reference["video_source"]!)
            let video = try H3ReferenceNormalizer.normalizeVideoFrames(
                videoSource, fps: 30, numFrames: numFrames)
            try compare(
                ours: frameStack(video),
                reference: reference["video"]!.asType(.float32),
                name: "video 30->24 fps + canvas (uint8)")

            // 3. Image reference: its own 2048 short edge, no area cap.
            let picture = try H3ReferenceNormalizer.normalizeImage(
                keyframeImage(from: reference["image_source"]!))
            try compare(
                ours: MLXArray(picture.pixels, [picture.height, picture.width, 3]).asType(.float32),
                reference: reference["image"]!.asType(.float32),
                name: "image reference 2048 short edge (uint8)")

            // 4. Soundtrack: truncate at 44.1 kHz, then one resample onto 32 kHz.
            let soundtrack = try H3ReferenceNormalizer.normalizeAudio(
                audioBuffer(from: reference["soundtrack_source"]!, sampleRate: 44_100),
                maxDuration: Double(numFrames) / 24.0)
            try compare(
                ours: audioArray(soundtrack), reference: reference["soundtrack"]!,
                name: "soundtrack 44.1 kHz -> 32 kHz")

            // 5. Mono audio reference at 16 kHz: upmixed by channel repeat, then upsampled.
            let mono = try H3ReferenceNormalizer.normalizeAudio(
                audioBuffer(from: reference["mono_source"]!, sampleRate: 16_000),
                maxDuration: Double(numFrames) / 24.0)
            try compare(
                ours: audioArray(mono), reference: reference["mono"]!,
                name: "mono reference 16 kHz -> 32 kHz stereo")

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

    /// Rebuild the synthetic `[audio, video+soundtrack, image]` request the ref2va dumps are
    /// built from, and prepare its vision blocks. Shared by the two ref2va conditioner probes so
    /// they cannot drift apart.
    private func ref2vaSyntheticPresentation(
        from reference: [String: MLXArray], modelDirectory: URL
    ) async throws -> (H3Presentation, [(patches: MLXArray, gridH: Int, gridW: Int)]) {
        let references: [H3Reference] = [
            .audio(H3AudioReference(
                audio: audioBuffer(from: reference["voice"]!, sampleRate: 16_000))),
            .video(H3VideoReference(
                frames: frames(from: reference["clip"]!), fps: 24,
                audio: audioBuffer(from: reference["clip_audio"]!, sampleRate: 32_000))),
            .image(H3ImageReference(image: keyframeImage(from: reference["picture"]!))),
        ]
        let normalized = try H3ReferenceNormalizer.normalize(references, numFrames: 124)

        var presentationReferences = [H3PresentationReference]()
        var visionBlocks = [(patches: MLXArray, gridH: Int, gridW: Int)]()
        for entry in normalized {
            switch entry {
            case .audio:
                presentationReferences.append(.audio)
            case .video(let video):
                let prepared = try H3ReferenceVideoProcessor.prepare(frames: video.frames)
                for block in prepared.blocks {
                    visionBlocks.append((block, prepared.gridH, prepared.gridW))
                }
                presentationReferences.append(.videoWithAudio(
                    gridH: prepared.gridH, gridW: prepared.gridW,
                    blockTimestamps: prepared.blockTimestamps))
            case .image(let entry):
                let (patches, gridH, gridW) = entry.image.visionPatches()
                visionBlocks.append((patches, gridH, gridW))
                presentationReferences.append(.image(gridH: gridH, gridW: gridW))
            }
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelDirectory.appendingPathComponent("tokenizer"))
        let presentation = try H3Presentation.ref2va(
            prompt: "subject_definitions:\n<Subject 1> is the room in <Picture 1>.\n\n"
                + "summary:\n[reference generation + audio reference] A short handheld take "
                + "in <Subject 1>.",
            references: presentationReferences, tokenizer: tokenizer)
        return (presentation, visionBlocks)
    }

    /// (T, H, W, 3) uint8 reference tensor -> per-frame bitmaps.
    private func frames(from array: MLXArray) -> [H3KeyframeImage] {
        let bytes = array.asType(.uint8).asArray(UInt8.self)
        let (count, height, width) = (array.dim(0), array.dim(1), array.dim(2))
        let stride = height * width * 3
        return (0..<count).map { index in
            H3KeyframeImage(
                width: width, height: height,
                pixels: Array(bytes[(index * stride)..<((index + 1) * stride)]))
        }
    }

    /// Bitmaps -> a (T, H, W, 3) float32 array for comparison.
    private func frameStack(_ frames: [H3KeyframeImage]) -> MLXArray {
        let pixels = frames.flatMap(\.pixels)
        return MLXArray(pixels, [frames.count, frames[0].height, frames[0].width, 3]).asType(.float32)
    }

    /// (channels, samples) float32 reference tensor -> waveform buffer.
    private func audioBuffer(from array: MLXArray, sampleRate: Int) -> H3AudioBuffer {
        H3AudioBuffer(
            channels: array.dim(0), frameCount: array.dim(1),
            samples: array.asType(.float32).asArray(Float.self), sampleRate: sampleRate)
    }

    private func audioArray(_ buffer: H3AudioBuffer) -> MLXArray {
        MLXArray(buffer.samples, [buffer.channels, buffer.frameCount])
    }

    private func compare(
        ours: MLXArray, reference: MLXArray, name: String, tolerance override: Float? = nil
    ) throws {
        let limit = override ?? tolerance
        guard ours.shape == reference.shape else {
            throw H3Error.generationFailed("\(name): shape \(ours.shape) vs reference \(reference.shape)")
        }
        let difference = abs(ours.asType(.float32) - reference.asType(.float32))
        let maxError = difference.max().item(Float.self)
        let meanError = mean(difference).item(Float.self)
        let refScale = abs(reference.asType(.float32)).max().item(Float.self)
        print("\(name): max|Δ| \(maxError)  mean|Δ| \(meanError)  (ref scale \(refScale))")
        guard maxError <= limit else {
            throw H3Error.generationFailed("\(name): max error \(maxError) exceeds tolerance \(limit)")
        }
        print("PARITY OK")
    }
}
