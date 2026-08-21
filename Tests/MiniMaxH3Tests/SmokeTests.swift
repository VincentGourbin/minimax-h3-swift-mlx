// SmokeTests.swift - checkpoint-backed smoke tests, skipped when the weights are absent
// Copyright 2026 Vincent Gourbin
//
// The suite in KeyframeConditioningTests.swift proves the geometry without a checkpoint. These
// prove the parts that only real weights can: that the loaders' key remapping still matches the
// released tensors, and that the small components produce finite, correctly-shaped output.
//
// They run only when `$H3_MODELS_DIR` points at a checkpoint, so a contributor without the
// 163 GB download still gets a green suite. Deliberately limited to the cheap components — the
// audio VAE (0.6 GB), the video VAE and the vision tower — so the whole thing stays under a
// minute and never approaches the memory ceiling. The 33B transformer and the 52 GB conditioner
// belong to `minimax-h3 smoke all`, not to a test suite.
//
// `xcodebuild test` does NOT forward the shell environment to the test process, so it always
// skips this suite. Run the built bundle directly to exercise it:
//
//   xcodebuild build-for-testing -scheme minimax-h3 -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild
//   H3_MODELS_DIR=/path/to/MiniMax-H3 \
//     xcrun xctest .xcodebuild/Build/Products/Debug/MiniMaxH3Tests.xctest

import Foundation
import MLX
import Testing
import Tokenizers
@testable import MiniMaxH3

/// The checkpoint directory, or nil when the environment does not point at one.
private func checkpointDirectory() -> URL? {
    guard let path = ProcessInfo.processInfo.environment["H3_MODELS_DIR"], !path.isEmpty else {
        return nil
    }
    let url = URL(fileURLWithPath: path)
    // Require a component we actually load, so a stale variable skips instead of failing.
    guard FileManager.default.fileExists(atPath: url.appendingPathComponent("audio_vae").path) else {
        return nil
    }
    return url
}

private let checkpoint = checkpointDirectory()

@Suite(.enabled(if: checkpoint != nil, "set H3_MODELS_DIR to run checkpoint-backed smoke tests"))
struct CheckpointSmokeTests {
    private let directory = checkpoint!

    @Test func audioVAEDecodesToStereoWaveform() throws {
        let vae = try H3WeightLoader.loadAudioVAE(modelDirectory: directory)
        // Two batch items = the two stereo channels the mono VAE decodes separately.
        let latents = MLXArray.zeros([2, 32, 8]).asType(.float32)
        let waveform = try vae.decode(latents)
        eval(waveform)
        #expect(waveform.dim(0) == 2)
        // hop 800 at 32 kHz: 8 latents -> 6400 samples per channel.
        #expect(waveform.dim(1) == 8 * 800)
        #expect(waveform.asType(.float32).abs().max().item(Float.self) < .infinity)
        #expect(vae.config.samplingRate == H3Constants.audioSampleRate)
    }

    @Test func videoVAERoundTripsAKeyframeThroughEncoderAndDecoder() throws {
        let vae = try H3WeightLoader.loadVideoVAE(modelDirectory: directory, includeEncoder: true)
        // One 64x64 frame — the smallest thing the tiled spatial encoder will take.
        let pixels = MLXArray.zeros([1, 3, 1, 64, 64]).asType(.float32)
        let moments = try vae.encodeClip(pixels)
        eval(moments)
        // Moments carry mean and log-variance: 2 x latentChannels, spatially compressed by 16.
        #expect(moments.dim(1) == 2 * vae.config.latentChannels)
        #expect(moments.dim(3) == 4 && moments.dim(4) == 4)
        #expect(moments.asType(.float32).abs().max().item(Float.self) < .infinity)
        #expect(vae.config.latentsMean.count == vae.config.latentChannels)
        #expect(vae.config.latentsStd.count == vae.config.latentChannels)
    }

    /// The tower must load in bf16 by default: running it in fp32 is more accurate in isolation
    /// but steers the conditioner away from the released trajectory (see `parity
    /// conditioner-fl2va`, which measured cosine 0.67 versus 0.9997).
    @Test func visionTowerDefaultsToBF16AndEmitsMergedTokensPlusDeepstack() throws {
        let tower = try H3WeightLoader.loadVisionTower(modelDirectory: directory)
        #expect(tower.patchEmbed.weight.dtype == .bfloat16)

        // A 64x64 keyframe -> 4x4 patches -> 4 merged tokens.
        let image = H3KeyframeImage(
            width: 64, height: 64, pixels: [UInt8](repeating: 128, count: 64 * 64 * 3))
        let (patches, gridH, gridW) = image.visionPatches()
        let (embeds, deepstack) = tower(patches, gridH: gridH, gridW: gridW)
        eval([embeds] + deepstack)
        #expect(embeds.shape == [4, tower.config.outHiddenSize])
        #expect(deepstack.count == tower.config.deepstackVisualIndexes.count)
        #expect(deepstack.allSatisfy { $0.shape == [4, tower.config.outHiddenSize] })
        #expect(embeds.asType(.float32).abs().max().item(Float.self) < .infinity)
    }

    /// The presentation must agree with the real tokenizer: one pad per merged token, and the
    /// mrope layout built from it must cover exactly the sequence.
    @Test func presentationMatchesTheRealTokenizer() async throws {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: directory.appendingPathComponent("tokenizer"))
        let presentation = try H3Presentation(
            prompt: "a red fox", imageGrids: [(h: 16, w: 28)], tokenizer: tokenizer)
        let layout = try #require(try presentation.multimodalLayout())
        let mergedTokens = (16 / 2) * (28 / 2)
        #expect(layout.imageRuns.map(\.count) == [mergedTokens])
        #expect(layout.positions.dim(1) == presentation.tokenIds.count)
        #expect(presentation.tokenTags.count == presentation.tokenIds.count)
        #expect(presentation.mmTokenTypes.count == presentation.tokenIds.count)
    }

    /// The ref2va presentation against the real tokenizer: labels numbered per modality in
    /// request order, `"<Audio j>: "` emitted BEFORE `"<Video k>: "` for a soundtrack-bearing
    /// video, one vision run per merged frame pair, and an mrope layout that covers the sequence.
    @Test func ref2vaPresentationLabelsAndRuns() async throws {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: directory.appendingPathComponent("tokenizer"))
        let presentation = try H3Presentation.ref2va(
            prompt: "subject_definitions:\n<Subject 1> is the room in <Picture 1>.",
            references: [
                .audio,
                .videoWithAudio(gridH: 8, gridW: 16, blockTimestamps: [0.25, 1.25]),
                .image(gridH: 16, gridW: 16),
            ],
            tokenizer: tokenizer)

        // Decoding without the pads shows the label structure the reference emits.
        let imagePad = try #require(tokenizer.convertTokenToId("<|image_pad|>"))
        let videoPad = try #require(tokenizer.convertTokenToId("<|video_pad|>"))
        let structural = presentation.tokenIds
            .filter { $0 != Int32(imagePad) && $0 != Int32(videoPad) }
            .map(Int.init)
        let decoded = tokenizer.decode(tokens: structural)
        #expect(decoded.hasPrefix("<Audio 1>: <Audio 2>: <Video 1>: <0.2 seconds>"))
        #expect(decoded.contains("<1.2 seconds>"))
        #expect(decoded.contains("<Picture 1>: "))

        // Three vision runs: two video blocks then the image.
        let layout = try #require(try presentation.multimodalLayout())
        #expect(layout.imageRuns.map(\.count) == [4 * 8, 4 * 8, 8 * 8])
        #expect(layout.positions.dim(1) == presentation.tokenIds.count)
        #expect(presentation.tokenTags.count == presentation.tokenIds.count)
        // A vision block's rows are tagged as VIDEO for H3's AdaLN, delimiters included, while the
        // mrope types mark only the pads (1 image, 2 video).
        #expect(Set(presentation.mmTokenTypes) == [0, 1, 2])
    }
}
