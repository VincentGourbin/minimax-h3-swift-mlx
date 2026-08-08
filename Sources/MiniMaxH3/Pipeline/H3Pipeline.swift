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
    /// nil -> canvas resolved from `aspectWidth:aspectHeight` (default 16:9, short edge 768).
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
    /// The vision tower (0.4B fp32) runs and is freed before the 52 GB text stack loads.
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

    // MARK: - Generation

    public func generate(_ request: H3GenerationRequest) async throws -> H3GenerationResult {
        // 1. Geometry.
        let (height, width): (Int, Int)
        if let h = request.height, let w = request.width {
            guard h % 32 == 0, w % 32 == 0 else {
                throw H3Error.invalidInput("height/width must be multiples of 32, got \(h)x\(w).")
            }
            (height, width) = (h, w)
        } else {
            (height, width) = try H3Geometry.resolveCanvasSize(
                aspectWidth: request.aspectWidth, aspectHeight: request.aspectHeight)
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

        // 2. Text conditioning, then free the encoder before anything big loads.
        let (promptEmbeds, textTags) = try await encodePrompt(
            request.prompt, quantization: request.textEncoderQuantization)
        Memory.clearCache()

        // 3. Packed layout and schedules.
        let layout = try H3Packing.buildPackedSequence(
            textTokenTags: textTags,
            numLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            numAudioLatents: audioLatents,
            patchSize: patch
        )
        H3Debug.log("packed sequence: \(layout.sequenceLength) rows")

        let videoScheduler = H3Scheduler(shift: request.flowShift)
        let audioScheduler = H3Scheduler(shift: request.audioFlowShift)
        try videoScheduler.setTimesteps(numInferenceSteps: request.numInferenceSteps)
        try audioScheduler.setTimesteps(numInferenceSteps: request.numInferenceSteps)
        let stepCount = min(videoScheduler.timesteps.count, audioScheduler.timesteps.count)

        // 4. Initial noise: video latent tensor first, then audio rows (draw order is part of the
        // request's reproducibility contract; torch and MLX RNGs differ, so seeds are MLX-local).
        MLXRandom.seed(request.seed)
        let videoNoise = MLXRandom.normal([1, 24, latentFrames, latentHeight, latentWidth], type: Float.self)
        var videoRows = H3Packing.patchifyVideoLatents(videoNoise, patchSize: patch)
        var audioRows = MLXRandom.normal([audioLatents * H3Constants.audioChannels, 32], type: Float.self)
        eval(videoRows, audioRows)

        // 5. Denoise. One forward per step (guidance-distilled).
        let profiler = H3Profiler.shared
        report("Loading transformer (61.7 GB)")
        profiler.start("Load Transformer")
        var transformer: H3Transformer? = try H3WeightLoader.loadTransformer(
            modelDirectory: modelDirectory, quantization: request.transformerQuantization)
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
            let (videoVelocity, audioVelocity) = transformer!(
                videoRows: videoRows,
                audioRows: audioRows,
                textEmbeds: promptEmbeds,
                timesteps: timesteps,
                timestepIndices: timestepIndices,
                layout: layout
            )
            videoRows = try videoScheduler.step(
                modelOutput: videoVelocity, timestep: videoT, sample: videoRows)
            audioRows = try audioScheduler.step(
                modelOutput: audioVelocity, timestep: audioT, sample: audioRows)
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
        let waveform = audioVAE.decode(audioLatentTensor)  // (2, samples)
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
