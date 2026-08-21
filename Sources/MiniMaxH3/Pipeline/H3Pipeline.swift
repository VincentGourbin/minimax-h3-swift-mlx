// H3Pipeline.swift - t2va generation pipeline with strict sequential stage loading
// Copyright 2026 Vincent Gourbin
//
// Stage plan for 96 GB (full precision): text encoder (~52 GB, layers 0..<50) -> UNLOAD ->
// transformer (61.7 GB) -> UNLOAD -> video VAE (10.4 GB, tiled+chunked decode) -> UNLOAD ->
// audio VAE (0.6 GB). Never two large components resident. `Memory.clearCache()` between stages.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

public struct H3GenerationRequest: Sendable {
    public var prompt: String
    /// fl2va: keyframe the video starts from. Stretched onto the canvas, which by default
    /// adopts this image's aspect ratio.
    public var image: H3KeyframeImage?
    /// fl2va: keyframe the video ends on. Alone it anchors "last" (and sets the canvas aspect);
    /// combined with `image` it follows that canvas and is cover-cropped onto it.
    public var lastImage: H3KeyframeImage?
    /// ref2va: the ordered reference list — images, videos (with their soundtrack) and audio
    /// clips. **The order is the request**: it numbers the `<Picture i>` / `<Audio j>` /
    /// `<Video k>` labels AND lays the blocks out on the shared rotary clock. A non-empty list
    /// switches the request to ref2va, which runs against `transformer_ref`.
    public var references: [H3Reference] = []
    /// nil -> canvas resolved from the first keyframe's aspect ratio, else from
    /// `aspectWidth:aspectHeight` (default 16:9, short edge 768). References never bind it.
    public var height: Int?
    public var width: Int?
    public var aspectWidth: Double = 16
    public var aspectHeight: Double = 9
    public var numFrames = 124
    /// Sigma grid points; drives numInferenceSteps - 1 transformer evaluations.
    public var numInferenceSteps = 50
    public var seed: UInt64 = 0
    public var flowShift: Float = 12.0
    public var audioFlowShift: Float = 3.0
    /// Skip the 5-15 s duration check (frames still snapped to 17n+5) — smoke tests only.
    public var allowShortVideo = false
    /// On-the-fly quantization of the 33B transformer.
    public var transformerQuantization: H3Quantization = .none
    /// On-the-fly quantization of the Qwen3-VL conditioner.
    public var textEncoderQuantization: H3Quantization = .none
    /// Graph-compile each transformer block (`MLX.compile`): same math, fused elementwise glue.
    public var compileBlocks = false

    /// Step-distillation LoRA folded into the transformer before quantization. The published
    /// Turbo adapters expect 4 or 8 transformer evaluations, i.e. `numInferenceSteps` 5 or 9.
    public var turboLoRA: URL?
    /// Extra multiplier on the folded delta; the LoRA cards use this as a sharpness dial.
    public var turboLoRAStrength: Float = 1.0
    /// Block-sparse attention keep fraction (nil = full attention). Approximation, opt-in.
    public var sparseAttentionKeep: Float?

    public init(prompt: String) { self.prompt = prompt }
}

public struct H3GenerationResult {
    /// (frames, height, width, 3) uint8 RGB.
    public let videoFrames: MLXArray
    /// (2, samples) float32 in [-1, 1] at 32 kHz.
    public let audioWaveform: MLXArray
    public let fps: Int
    public let sampleRate: Int
}

public final class H3Pipeline {
    public let modelDirectory: URL
    public var progressHandler: (@Sendable (String, Int, Int) -> Void)?

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    private func report(_ phase: String, _ step: Int = 0, _ total: Int = 0) {
        progressHandler?(phase, step, total)
        H3Debug.log("\(phase)\(total > 0 ? " \(step)/\(total)" : "")")
    }

    // MARK: - Stage 1: text encoding

    /// Encode the presentation: the verbatim prompt for t2va; per-keyframe `"<Picture i>: "`
    /// labels + vision blocks first for fl2va (keyframes already prepared onto the canvas).
    /// The vision tower (0.4B, bf16 to match the release — see `loadVisionTower`) runs and is
    /// freed before the 52 GB text stack loads.
    func encodePrompt(
        _ prompt: String, keyframes: [H3KeyframeImage] = [], quantization: H3Quantization
    ) async throws -> (embeds: MLXArray, tags: [Int32]) {
        let profiler = H3Profiler.shared
        report("Loading tokenizer")
        profiler.start("Tokenization")
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelDirectory.appendingPathComponent("tokenizer")
        )
        profiler.end("Tokenization")

        var imageEmbeds: MLXArray?
        var deepstack = [MLXArray]()
        var imageGrids = [(h: Int, w: Int)]()
        if !keyframes.isEmpty {
            report("Encoding \(keyframes.count) keyframe(s) (vision tower)")
            profiler.start("Vision Tower")
            var tower: Qwen3VLVisionTower? = try H3WeightLoader.loadVisionTower(
                modelDirectory: modelDirectory)
            var embedsPerImage = [MLXArray]()
            var deepstackPerImage = [[MLXArray]]()
            for keyframe in keyframes {
                let (patches, gridH, gridW) = keyframe.visionPatches()
                let (embeds, taps) = tower!(patches, gridH: gridH, gridW: gridW)
                embedsPerImage.append(embeds)
                deepstackPerImage.append(taps)
                imageGrids.append((gridH, gridW))
            }
            imageEmbeds = concatenated(embedsPerImage, axis: 0)
            deepstack = (0..<deepstackPerImage[0].count).map { level in
                concatenated(deepstackPerImage.map { $0[level] }, axis: 0)
            }
            eval([imageEmbeds!] + deepstack)
            tower = nil
            Memory.clearCache()
            profiler.end("Vision Tower")
        }

        let presentation = try H3Presentation(
            prompt: prompt, imageGrids: imageGrids, tokenizer: tokenizer)
        let layout = try presentation.multimodalLayout()

        report("Loading text encoder (Qwen3-VL-32B, layers 0-49)")
        profiler.start("Load Text Encoder")
        let encoder = try H3WeightLoader.loadTextEncoder(
            modelDirectory: modelDirectory, quantization: quantization)
        profiler.end("Load Text Encoder")

        report("Encoding prompt (\(presentation.tokenIds.count) tokens)")
        profiler.start("Text Encoding")
        let embeds = encoder(
            MLXArray(presentation.tokenIds).expandedDimensions(axis: 0),
            imageEmbeds: imageEmbeds,
            deepstack: deepstack,
            layout: layout
        )
        eval(embeds)
        profiler.end("Text Encoding")
        return (embeds, presentation.tokenTags)
    }

    // MARK: - Stage 1b: keyframe VAE encoding (fl2va)

    /// Encode prepared keyframes into packed conditioning rows (before noise augmentation).
    ///
    /// Reference contract (`MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes`): single frames go
    /// through the spatial encoder alone; the posterior is *sampled* under a fresh seed-42
    /// generator per keyframe (independent of the request seed; torch and MLX RNGs differ, so
    /// the draw is MLX-local); the sampled latent is rounded through float16 BEFORE normalization
    /// — ~11 bits of every conditioning latent, part of the released model's conditioning.
    func encodeKeyframes(
        _ keyframes: [H3KeyframeImage], patchSize: (t: Int, h: Int, w: Int)
    ) throws -> MLXArray {
        let profiler = H3Profiler.shared
        report("Encoding \(keyframes.count) keyframe(s) (video VAE)")
        profiler.start("Keyframe VAE Encode")
        // Encoder only: this stage never decodes, and the ViT decoder is ~5 GB of dead weight
        // here — the decode stage loads its own copy once the transformer is gone.
        var vae: H3VideoVAE? = try H3WeightLoader.loadVideoVAE(
            modelDirectory: modelDirectory, includeEncoder: true, includeDecoder: false)
        let latentsMean = MLXArray(vae!.config.latentsMean).reshaped(1, -1, 1, 1, 1)
        let latentsStd = MLXArray(vae!.config.latentsStd).reshaped(1, -1, 1, 1, 1)
        let pixelMean = MLXArray(H3Constants.pixelMean).reshaped(1, -1, 1, 1, 1)
        let pixelStd = MLXArray(H3Constants.pixelStd).reshaped(1, -1, 1, 1, 1)

        var rows = [MLXArray]()
        for keyframe in keyframes {
            var pixels = MLXArray(keyframe.pixels, [keyframe.height, keyframe.width, 3])
                .asType(.float32)
                .transposed(2, 0, 1)
                .reshaped(1, 3, 1, keyframe.height, keyframe.width)
            pixels = (pixels / 255.0 - pixelMean) / pixelStd
            let moments = try vae!.encodeClip(pixels)  // (1, 2C, 1, hLat, wLat)
            let channels = moments.dim(1) / 2
            let mean = moments[0..., ..<channels]
            let logvar = clip(moments[0..., channels...], min: -30.0, max: 20.0)
            MLXRandom.seed(H3Constants.keyframeEncodeSeed)
            let sampled = mean + exp(0.5 * logvar) * MLXRandom.normal(mean.shape, type: Float.self)
            let latents = sampled.asType(.float16).asType(.float32)
            rows.append(
                H3Packing.patchifyVideoLatents((latents - latentsMean) / latentsStd, patchSize: patchSize))
        }
        let conditionRows = concatenated(rows, axis: 0)
        eval(conditionRows)
        vae = nil
        Memory.clearCache()
        profiler.end("Keyframe VAE Encode")
        return conditionRows
    }

    // MARK: - Stage 1c: ref2va conditioning

    /// Encode the `ref2va` presentation: one label per reference, numbered per modality in request
    /// order, then the prompt verbatim. Image references go through the vision tower at their own
    /// 2048-edge resolution; a video reference contributes one timestamped vision block per merged
    /// frame pair; an audio reference contributes a label and nothing else — a waveform never
    /// reaches the conditioner.
    ///
    /// Every vision block is an independent tower call. That is not an approximation: Qwen3-VL
    /// segments its vision attention per temporal group and its rotary table has no temporal axis,
    /// so a `grid_t = N` call and N `grid_t = 1` calls are the same tensor (checked against the
    /// checkpoint by the `video-condition` parity probe, which measured max|Δ| 0).
    func encodeRef2VAPrompt(
        _ prompt: String, references: [H3Reference], quantization: H3Quantization
    ) async throws -> (embeds: MLXArray, tags: [Int32]) {
        let profiler = H3Profiler.shared
        report("Loading tokenizer")
        profiler.start("Tokenization")
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelDirectory.appendingPathComponent("tokenizer"))
        profiler.end("Tokenization")

        // Prepare every reference's vision blocks first, so the tower is loaded once and freed
        // before the 52 GB text stack lands.
        var presentationReferences = [H3PresentationReference]()
        var visionBlocks = [(patches: MLXArray, gridH: Int, gridW: Int)]()
        for reference in references {
            switch reference {
            case .image(let entry):
                let (patches, gridH, gridW) = entry.image.visionPatches()
                visionBlocks.append((patches, gridH, gridW))
                presentationReferences.append(.image(gridH: gridH, gridW: gridW))
            case .video(let entry):
                let prepared = try H3ReferenceVideoProcessor.prepare(frames: entry.frames)
                for block in prepared.blocks {
                    visionBlocks.append((block, prepared.gridH, prepared.gridW))
                }
                presentationReferences.append(
                    entry.audio == nil
                        ? .video(
                            gridH: prepared.gridH, gridW: prepared.gridW,
                            blockTimestamps: prepared.blockTimestamps)
                        : .videoWithAudio(
                            gridH: prepared.gridH, gridW: prepared.gridW,
                            blockTimestamps: prepared.blockTimestamps))
            case .audio:
                presentationReferences.append(.audio)
            }
        }

        var imageEmbeds: MLXArray?
        var deepstack = [MLXArray]()
        if !visionBlocks.isEmpty {
            report("Encoding \(visionBlocks.count) reference vision block(s)")
            profiler.start("Vision Tower")
            var tower: Qwen3VLVisionTower? = try H3WeightLoader.loadVisionTower(
                modelDirectory: modelDirectory)
            var embedsPerBlock = [MLXArray]()
            var deepstackPerBlock = [[MLXArray]]()
            for block in visionBlocks {
                let (embeds, taps) = tower!(block.patches, gridH: block.gridH, gridW: block.gridW)
                embedsPerBlock.append(embeds)
                deepstackPerBlock.append(taps)
                eval([embeds] + taps)
            }
            imageEmbeds = concatenated(embedsPerBlock, axis: 0)
            deepstack = (0..<deepstackPerBlock[0].count).map { level in
                concatenated(deepstackPerBlock.map { $0[level] }, axis: 0)
            }
            eval([imageEmbeds!] + deepstack)
            tower = nil
            Memory.clearCache()
            profiler.end("Vision Tower")
        }

        let presentation = try H3Presentation.ref2va(
            prompt: prompt, references: presentationReferences, tokenizer: tokenizer)
        let layout = try presentation.multimodalLayout()

        report("Loading text encoder (Qwen3-VL-32B, layers 0-49)")
        profiler.start("Load Text Encoder")
        let encoder = try H3WeightLoader.loadTextEncoder(
            modelDirectory: modelDirectory, quantization: quantization)
        profiler.end("Load Text Encoder")

        report("Encoding presentation (\(presentation.tokenIds.count) tokens)")
        profiler.start("Text Encoding")
        let embeds = encoder(
            MLXArray(presentation.tokenIds).expandedDimensions(axis: 0),
            imageEmbeds: imageEmbeds,
            deepstack: deepstack,
            layout: layout)
        eval(embeds)
        profiler.end("Text Encoding")
        return (embeds, presentation.tokenTags)
    }

    /// What the reference encoders produced, in packed order.
    struct H3EncodedReferences {
        /// One `(1, 24, F, H, W)` normalized conditioning latent per image and video reference.
        var conditionLatents: [MLXArray]
        /// One `(latents * 2, 32)` channel-major row block per audio-bearing reference.
        var audioRows: [MLXArray]
        /// The block geometry the packed layout is built from.
        var blocks: [H3ReferenceBlockGeometry]
    }

    /// Encode the references: images and videos through the video VAE, soundtracks and audio
    /// references through the audio VAE.
    ///
    /// The visual recipe is the keyframe one (`encode_vae_condition`): ImageNet-normalized pixels,
    /// a posterior *sampled* under a fresh seed-42 generator per reference, the sample rounded
    /// through float16 BEFORE normalization. Soundtracks take the posterior **mean** and are never
    /// sampled — no seed is involved in conditioning on audio.
    func encodeReferences(
        _ references: [H3Reference], patchSize: (t: Int, h: Int, w: Int)
    ) throws -> H3EncodedReferences {
        let profiler = H3Profiler.shared
        report("Encoding \(references.count) reference(s) (VAEs)")
        profiler.start("Reference VAE Encode")

        var conditionLatents = [MLXArray]()
        var audioRows = [MLXArray]()
        var blocks = [H3ReferenceBlockGeometry]()

        // Visual side first, then the audio VAE: never two model stacks resident.
        if references.contains(where: { $0.kind != .audio }) {
            var vae: H3VideoVAE? = try H3WeightLoader.loadVideoVAE(
                modelDirectory: modelDirectory, includeEncoder: true, includeDecoder: false)
            let latentsMean = MLXArray(vae!.config.latentsMean).reshaped(1, -1, 1, 1, 1)
            let latentsStd = MLXArray(vae!.config.latentsStd).reshaped(1, -1, 1, 1, 1)
            let pixelMean = MLXArray(H3Constants.pixelMean).reshaped(1, -1, 1, 1, 1)
            let pixelStd = MLXArray(H3Constants.pixelStd).reshaped(1, -1, 1, 1, 1)

            for reference in references {
                let frames: [H3KeyframeImage]
                switch reference {
                case .audio: continue
                case .image(let entry): frames = [entry.image]
                case .video(let entry):
                    // Snap *down* to `17n + 5` so the VAE encodes without padding. `max(1, ...)`
                    // can nominate more frames than the reference carries, which the reference
                    // implementation resolves by slicing — so does this.
                    let count = entry.frames.count
                    let chunks = max(1, floorDivide(count - H3Constants.latentsPerChunk, H3Constants.framesPerChunk))
                    let wanted = chunks * H3Constants.framesPerChunk + H3Constants.latentsPerChunk
                    frames = Array(entry.frames.prefix(wanted))
                }
                let (height, width) = (frames[0].height, frames[0].width)
                var pixels = MLXArray(frames.flatMap(\.pixels), [frames.count, height, width, 3])
                    .asType(.float32)
                    .transposed(3, 0, 1, 2)
                    .expandedDimensions(axis: 0)  // (1, 3, T, H, W)
                pixels = (pixels / 255.0 - pixelMean) / pixelStd
                let moments = try vae!.encodeVideo(pixels)
                let channels = moments.dim(1) / 2
                let mean = moments[0..., ..<channels]
                let logvar = clip(moments[0..., channels...], min: -30.0, max: 20.0)
                MLXRandom.seed(H3Constants.keyframeEncodeSeed)
                let sampled = mean + exp(0.5 * logvar) * MLXRandom.normal(mean.shape, type: Float.self)
                let latents = sampled.asType(.float16).asType(.float32)
                let normalized = (latents - latentsMean) / latentsStd
                eval(normalized)
                conditionLatents.append(normalized)
            }
            vae = nil
            Memory.clearCache()
        }

        if references.contains(where: \.hasAudio) {
            let vae = try H3WeightLoader.loadAudioVAE(
                modelDirectory: modelDirectory, includeEncoder: true, includeDecoder: false)
            let mean = MLXArray(vae.config.latentsMean).reshaped(1, 1, -1)
            let std = MLXArray(vae.config.latentsStd).reshaped(1, 1, -1)
            for reference in references {
                guard let buffer = reference.audioBuffer else { continue }
                // Stereo as two batch items of the mono VAE; the posterior MODE, never a sample.
                let waveform = MLXArray(buffer.samples, [buffer.channels, buffer.frameCount])
                let (posteriorMean, _) = try vae.encode(waveform)
                let latents = posteriorMean.transposed(0, 2, 1)  // (2, T, 32)
                let normalized = ((latents - mean) / std).reshaped(-1, vae.config.latentChannels)
                eval(normalized)
                audioRows.append(normalized)
            }
            Memory.clearCache()
        }

        // Walk the references once more to assemble the geometry, consuming the two lists in the
        // order they were filled — they skip the references they do not apply to.
        var nextVisual = 0
        var nextAudio = 0
        for reference in references {
            var block = H3ReferenceBlockGeometry(kind: reference.kind)
            if reference.kind != .audio {
                let latents = conditionLatents[nextVisual]
                nextVisual += 1
                block.latentFrames = latents.dim(2)
                block.latentHeight = latents.dim(3)
                block.latentWidth = latents.dim(4)
                // Caught here rather than 50 layers deep: a latent grid the patch does not divide
                // surfaces in the reference as an `index_copy` shape error inside the transformer,
                // which says nothing about which reference caused it.
                guard block.latentHeight % patchSize.h == 0, block.latentWidth % patchSize.w == 0
                else {
                    throw H3Error.invalidInput(
                        "Reference \(blocks.count + 1) encodes to "
                            + "\(block.latentHeight)x\(block.latentWidth) latents, which the "
                            + "\(patchSize.h)x\(patchSize.w) patch does not divide.")
                }
            }
            if reference.hasAudio {
                block.audioRows = audioRows[nextAudio].dim(0)
                nextAudio += 1
            }
            blocks.append(block)
        }
        profiler.end("Reference VAE Encode")
        return H3EncodedReferences(
            conditionLatents: conditionLatents, audioRows: audioRows, blocks: blocks)
    }

    /// Python's floor division, which the reference's `(T - 5) // 17` relies on for short clips.
    private func floorDivide(_ numerator: Int, _ denominator: Int) -> Int {
        let quotient = numerator / denominator
        return (numerator % denominator != 0 && (numerator < 0) != (denominator < 0))
            ? quotient - 1 : quotient
    }

    // MARK: - Generation



    public func generate(_ request: H3GenerationRequest) async throws -> H3GenerationResult {
        // Mutable so the raw reference media can be dropped the moment it has been normalized:
        // decoded frames are held at their SOURCE resolution, and a 4K clip is gigabytes that
        // must not still be resident when the 52 GB conditioner lands.
        var request = request
        // 1. Geometry. The first keyframe (if any) is the geometry anchor: it sets the canvas
        // aspect and is stretched onto it; a second keyframe follows and is cover-cropped.
        var keyframes = [H3KeyframeImage]()
        var keyframeAnchors = [String]()
        if let image = request.image {
            keyframes.append(image)
            keyframeAnchors.append("first")
        }
        if let lastImage = request.lastImage {
            keyframes.append(lastImage)
            keyframeAnchors.append("last")
        }

        let isRef2VA = !request.references.isEmpty
        guard !isRef2VA || keyframes.isEmpty else {
            throw H3Error.invalidInput(
                "A request is either `fl2va` (keyframes) or `ref2va` (references), not both.")
        }

        let (height, width): (Int, Int)
        if let h = request.height, let w = request.width {
            guard h % 32 == 0, w % 32 == 0 else {
                throw H3Error.invalidInput("height/width must be multiples of 32, got \(h)x\(w).")
            }
            (height, width) = (h, w)
        } else if let anchor = keyframes.first {
            (height, width) = try H3Geometry.resolveCanvasSize(
                aspectWidth: Double(anchor.width), aspectHeight: Double(anchor.height))
        } else {
            (height, width) = try H3Geometry.resolveCanvasSize(
                aspectWidth: request.aspectWidth, aspectHeight: request.aspectHeight)
        }
        keyframes = keyframes.enumerated().map { index, keyframe in
            keyframe.prepared(canvasWidth: width, canvasHeight: height, stretch: index == 0)
        }
        let numFrames = try H3Geometry.alignNumFrames(request.numFrames)
        let duration = Double(numFrames) / Double(H3Constants.fps)
        if !request.allowShortVideo,
           !(H3Constants.minDuration...H3Constants.maxDuration).contains(duration) {
            throw H3Error.invalidInput(
                "Duration \(duration)s outside 5-15 s (frames snapped to \(numFrames)). "
                    + "Use allowShortVideo for smoke tests.")
        }
        let latentFrames = try H3Geometry.videoLatentNumFrames(numFrames)
        let spatialRatio = 16
        let (latentHeight, latentWidth) = (height / spatialRatio, width / spatialRatio)
        let audioLatents = H3Geometry.audioLatentNumFrames(numFrames)
        let patch = (t: 1, h: 2, w: 2)
        H3Debug.log(
            "canvas \(width)x\(height), \(numFrames) frames -> latents \(latentFrames)x\(latentHeight)x\(latentWidth), "
                + "\(audioLatents) audio latents/channel")

        // 1b. ref2va: normalize the references onto H3's own rates and resolutions. This has to
        // happen after the frame count is final — a soundtrack is truncated to the generated
        // duration.
        let references = isRef2VA
            ? try H3ReferenceNormalizer.normalize(request.references, numFrames: numFrames)
            : []
        // The normalized copy is the only one anything downstream reads; the source-resolution
        // frames are dead weight from here on and the next stage is the largest one in the run.
        request.references = []

        // 2. Text conditioning (vision tower first for fl2va / ref2va), then free the encoder
        // before anything big loads.
        let (promptEmbeds, textTags) = isRef2VA
            ? try await encodeRef2VAPrompt(
                request.prompt, references: references,
                quantization: request.textEncoderQuantization)
            : try await encodePrompt(
                request.prompt, keyframes: keyframes,
                quantization: request.textEncoderQuantization)
        Memory.clearCache()

        // 2b. Conditioning latents (before noise augmentation).
        var conditionRows: MLXArray?
        var encodedReferences: H3EncodedReferences?
        let layout: H3PackedSequence
        if isRef2VA {
            let encoded = try encodeReferences(references, patchSize: patch)
            encodedReferences = encoded
            layout = try H3Packing.buildRef2VAPackedSequence(
                textTokenTags: textTags,
                blocks: encoded.blocks,
                numLatentFrames: latentFrames,
                latentHeight: latentHeight,
                latentWidth: latentWidth,
                numAudioLatents: audioLatents,
                patchSize: patch)
        } else {
            conditionRows = keyframes.isEmpty ? nil : try encodeKeyframes(keyframes, patchSize: patch)
            // 3. Packed layout and schedules.
            layout = try H3Packing.buildPackedSequence(
                textTokenTags: textTags,
                numLatentFrames: latentFrames,
                latentHeight: latentHeight,
                latentWidth: latentWidth,
                numAudioLatents: audioLatents,
                patchSize: patch,
                keyframeAnchors: keyframeAnchors
            )
        }
        if let encoded = encodedReferences {
            for (index, block) in encoded.blocks.enumerated() {
                H3Debug.log(
                    "reference \(index + 1) (\(block.kind.rawValue)): "
                        + (block.kind == .audio
                            ? "\(block.audioRows / H3Constants.audioChannels) audio latents"
                            : "\(block.latentFrames)x\(block.latentHeight)x\(block.latentWidth) latents"
                                + (block.audioRows > 0
                                    ? " + \(block.audioRows / H3Constants.audioChannels) audio latents" : "")))
            }
        }
        H3Debug.log("packed sequence: \(layout.sequenceLength) rows "
            + "(\(layout.numConditionVideoRows) condition video, "
            + "\(layout.numConditionAudioRows) condition audio)")

        let videoScheduler = H3Scheduler(shift: request.flowShift)
        let audioScheduler = H3Scheduler(shift: request.audioFlowShift)
        try videoScheduler.setTimesteps(numInferenceSteps: request.numInferenceSteps)
        try audioScheduler.setTimesteps(numInferenceSteps: request.numInferenceSteps)
        let stepCount = min(videoScheduler.timesteps.count, audioScheduler.timesteps.count)

        // 4. Noise, off the request seed. Draw order is part of the reproducibility contract
        // (torch and MLX RNGs differ, so seeds are MLX-local): condition noise per keyframe
        // FIRST, then the video latent tensor, then the audio rows. Condition rows are mixed to
        // the constant conditioning level (`scale_noise`: t·x0 + (1−t)·noise at t = 0.999) once,
        // and never touched again — they anchor the whole denoising loop.
        MLXRandom.seed(request.seed)
        let aug = H3Constants.keyframeNoiseAug
        if let encoded = encodedReferences {
            // One draw per VISUAL condition, in packed order, each at its own shape — `ref2va`
            // references are encoded at their own resolutions and never share one. Soundtracks are
            // never noised: a reference waveform conditions clean, at t = 1.0.
            var packed = [MLXArray]()
            for condition in encoded.conditionLatents {
                let noise = MLXRandom.normal(condition.shape, type: Float.self)
                let noised = aug * condition + (1 - aug) * noise
                packed.append(H3Packing.patchifyVideoLatents(noised, patchSize: patch))
            }
            if !packed.isEmpty {
                conditionRows = concatenated(packed, axis: 0)
                eval(conditionRows!)
                guard conditionRows!.dim(0) == layout.numConditionVideoRows else {
                    throw H3Error.generationFailed(
                        "The layout reserved \(layout.numConditionVideoRows) conditioning rows but the "
                            + "encoded references pack into \(conditionRows!.dim(0)).")
                }
            }
        } else if let rows = conditionRows {
            var noiseRows = [MLXArray]()
            for _ in keyframeAnchors {
                let noise = MLXRandom.normal([1, 24, 1, latentHeight, latentWidth], type: Float.self)
                noiseRows.append(H3Packing.patchifyVideoLatents(noise, patchSize: patch))
            }
            let noise = concatenated(noiseRows, axis: 0)
            conditionRows = aug * rows + (1 - aug) * noise
            eval(conditionRows!)
        }
        let videoNoise = MLXRandom.normal([1, 24, latentFrames, latentHeight, latentWidth], type: Float.self)
        var videoRows = H3Packing.patchifyVideoLatents(videoNoise, patchSize: patch)
        var audioRows = MLXRandom.normal([audioLatents * H3Constants.audioChannels, 32], type: Float.self)
        // ref2va: the reference soundtracks sit in front of the generated audio rows and ride
        // through every step unchanged, exactly as the visual conditioning does on the video side.
        // Kept apart from `audioRows`, which stays the generated rows the scheduler steps.
        let referenceAudioBlocks = encodedReferences.map(\.audioRows) ?? []
        var audioConditionRows: MLXArray?
        if !referenceAudioBlocks.isEmpty {
            let rows = referenceAudioBlocks.count == 1
                ? referenceAudioBlocks[0] : concatenated(referenceAudioBlocks, axis: 0)
            guard rows.dim(0) == layout.numConditionAudioRows else {
                throw H3Error.generationFailed(
                    "The layout reserved \(layout.numConditionAudioRows) reference audio rows but the "
                        + "encoded soundtracks pack into \(rows.dim(0)).")
            }
            audioConditionRows = rows
            eval(rows)
        }
        eval(videoRows, audioRows)

        // 5. Denoise. One forward per step (guidance-distilled).
        let profiler = H3Profiler.shared
        // `ref2va` runs against its own partition. Its `config.json` is byte-identical to the main
        // transformer's, so the loader, the quantization filter and the prequantized export all
        // apply unchanged — the directory name is the only difference.
        let transformerComponent = isRef2VA ? "transformer_ref" : "transformer"
        report("Loading \(transformerComponent) (61.7 GB)")
        profiler.start("Load Transformer")
        var transformer: H3Transformer? = try H3WeightLoader.loadTransformer(
            modelDirectory: modelDirectory, quantization: request.transformerQuantization,
            turboLoRA: request.turboLoRA, turboLoRAStrength: request.turboLoRAStrength,
            component: transformerComponent)
        transformer!.compileBlocks = request.compileBlocks
        transformer!.sparseAttentionKeep = request.sparseAttentionKeep
        Memory.clearCache()
        profiler.end("Load Transformer")

        profiler.start("Denoising")
        profiler.setTotalSteps(stepCount)
        for step in 0..<stepCount {
            let stepStart = Date()
            let videoT = videoScheduler.timesteps[step]
            let audioT = audioScheduler.timesteps[step]
            let (timesteps, timestepIndices) = H3Packing.buildRowTimesteps(
                layout: layout,
                videoTimestep: videoT,
                audioTimestep: audioT,
                conditionVideoTimestep: max(videoT, H3Constants.keyframeNoiseAug),
                conditionAudioTimestep: 1.0
            )
            let packedVideoRows = conditionRows.map { concatenated([$0, videoRows], axis: 0) }
                ?? videoRows
            let packedAudioRows = audioConditionRows.map { concatenated([$0, audioRows], axis: 0) }
                ?? audioRows
            let (videoVelocity, audioVelocity) = transformer!(
                videoRows: packedVideoRows,
                audioRows: packedAudioRows,
                textEmbeds: promptEmbeds,
                timesteps: timesteps,
                timestepIndices: timestepIndices,
                layout: layout
            )
            // Only the generated rows step; the conditioning rows are never updated.
            let generatedVelocity = conditionRows == nil
                ? videoVelocity : videoVelocity[layout.numConditionVideoRows...]
            let generatedAudioVelocity = audioConditionRows == nil
                ? audioVelocity : audioVelocity[layout.numConditionAudioRows...]
            videoRows = try videoScheduler.step(
                modelOutput: generatedVelocity, timestep: videoT, sample: videoRows)
            audioRows = try audioScheduler.step(
                modelOutput: generatedAudioVelocity, timestep: audioT, sample: audioRows)
            eval(videoRows, audioRows)
            profiler.recordStep(duration: Date().timeIntervalSince(stepStart))
            if (step + 1) % 5 == 0 { Memory.clearCache() }
            report("Denoising", step + 1, stepCount)
        }
        profiler.end("Denoising")

        // Free the transformer before anything else loads — 61.7 GB must not coexist with the VAE.
        transformer = nil
        Memory.clearCache()

        // 6. Video decode.
        report("Loading video VAE")
        profiler.start("Load Video VAE")
        let videoVAE = try H3WeightLoader.loadVideoVAE(modelDirectory: modelDirectory)
        profiler.end("Load Video VAE")
        var latents = H3Packing.unpatchifyVideoTokens(
            videoRows,
            numLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            channels: 24,
            patchSize: patch
        ).asType(.float32)
        let vMean = MLXArray(videoVAE.config.latentsMean).reshaped(1, -1, 1, 1, 1)
        let vStd = MLXArray(videoVAE.config.latentsStd).reshaped(1, -1, 1, 1, 1)
        latents = latents * vStd + vMean

        report("Decoding video")
        profiler.start("VAE Decode")
        var video = videoVAE.decode(latents)  // (1, 3, F, H, W), ImageNet space
        let pMean = MLXArray(H3Constants.pixelMean).reshaped(1, -1, 1, 1, 1)
        let pStd = MLXArray(H3Constants.pixelStd).reshaped(1, -1, 1, 1, 1)
        video = clip(video.asType(.float32) * pStd + pMean, min: 0.0, max: 1.0)
        let frames = (video[0].transposed(1, 2, 3, 0) * 255.0).round().asType(.uint8)
        eval(frames)
        Memory.clearCache()
        profiler.end("VAE Decode")

        // 7. Audio decode (stereo = batch of 2 through the mono VAE).
        report("Decoding audio")
        profiler.start("Audio Decode")
        let audioVAE = try H3WeightLoader.loadAudioVAE(modelDirectory: modelDirectory)
        var audioLatentTensor = H3Packing.unpackAudioTokens(audioRows, numAudioLatents: audioLatents)
            .asType(.float32)
        let aMean = MLXArray(audioVAE.config.latentsMean).reshaped(1, -1, 1)
        let aStd = MLXArray(audioVAE.config.latentsStd).reshaped(1, -1, 1)
        audioLatentTensor = audioLatentTensor * aStd + aMean
        let waveform = try audioVAE.decode(audioLatentTensor)  // (2, samples)
        eval(waveform)
        Memory.clearCache()
        profiler.end("Audio Decode")

        return H3GenerationResult(
            videoFrames: frames,
            audioWaveform: waveform,
            fps: H3Constants.fps,
            sampleRate: audioVAE.config.samplingRate
        )
    }
}
