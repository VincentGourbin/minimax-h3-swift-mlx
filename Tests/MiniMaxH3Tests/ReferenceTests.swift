// ReferenceTests.swift - ref2va Phase A: ingestion, ordering, limits, normalization
// Copyright 2026 Vincent Gourbin
//
// Everything here runs without a checkpoint. The reference contracts under test come from
// diffusers `references.py` and `MiniMaxH3Ref2VASetupStep`.

import Foundation
import Testing
@testable import MiniMaxH3

/// A solid-colour bitmap, enough for every geometry assertion.
private func image(_ width: Int, _ height: Int, value: UInt8 = 128) -> H3KeyframeImage {
    H3KeyframeImage(
        width: width, height: height,
        pixels: [UInt8](repeating: value, count: width * height * 3))
}

/// A frame whose top-left pixel encodes its index, so ordering survives every transform.
private func numberedFrame(_ index: Int, width: Int = 64, height: Int = 32) -> H3KeyframeImage {
    var pixels = [UInt8](repeating: 7, count: width * height * 3)
    pixels[0] = UInt8(index % 256)
    return H3KeyframeImage(width: width, height: height, pixels: pixels)
}

private func silence(channels: Int, frameCount: Int, sampleRate: Int?) -> H3AudioBuffer {
    H3AudioBuffer(
        channels: channels, frameCount: frameCount,
        samples: [Float](repeating: 0, count: channels * frameCount), sampleRate: sampleRate)
}

@Suite struct ReferenceValidationTests {
    @Test func emptyListRejected() {
        #expect(throws: H3Error.self) { try H3ReferenceNormalizer.validate([]) }
    }

    @Test func audioOnlyRejected() {
        let audio = H3Reference.audio(H3AudioReference(audio: silence(channels: 2, frameCount: 100, sampleRate: 32_000)))
        #expect(throws: H3Error.self) { try H3ReferenceNormalizer.validate([audio, audio]) }
        // Paired with anything visual it is accepted.
        let picture = H3Reference.image(H3ImageReference(image: image(64, 64)))
        #expect(throws: Never.self) { try H3ReferenceNormalizer.validate([audio, picture]) }
    }

    @Test func perModalityLimits() {
        let picture = H3Reference.image(H3ImageReference(image: image(64, 64)))
        let clip = H3Reference.video(H3VideoReference(frames: [image(64, 64)], fps: 24))
        let sound = H3Reference.audio(H3AudioReference(audio: silence(channels: 1, frameCount: 10, sampleRate: nil)))

        #expect(throws: Never.self) { try H3ReferenceNormalizer.validate([H3Reference](repeating: picture, count: 9)) }
        #expect(throws: H3Error.self) { try H3ReferenceNormalizer.validate([H3Reference](repeating: picture, count: 10)) }
        #expect(throws: Never.self) { try H3ReferenceNormalizer.validate([H3Reference](repeating: clip, count: 3)) }
        #expect(throws: H3Error.self) { try H3ReferenceNormalizer.validate([H3Reference](repeating: clip, count: 4)) }
        #expect(throws: H3Error.self) {
            try H3ReferenceNormalizer.validate([H3Reference](repeating: sound, count: 4) + [picture])
        }
    }

    @Test func totalCapIsTwelve() {
        let picture = H3Reference.image(H3ImageReference(image: image(64, 64)))
        let clip = H3Reference.video(H3VideoReference(frames: [image(64, 64)], fps: 24))
        let sound = H3Reference.audio(H3AudioReference(audio: silence(channels: 2, frameCount: 10, sampleRate: nil)))
        // 9 + 3 = 12 passes; one more of any kind does not.
        let full = [H3Reference](repeating: picture, count: 9) + [H3Reference](repeating: clip, count: 3)
        #expect(throws: Never.self) { try H3ReferenceNormalizer.validate(full) }
        #expect(throws: H3Error.self) { try H3ReferenceNormalizer.validate(full + [sound]) }
    }

    @Test func orderIsPreservedAcrossNormalization() throws {
        let references: [H3Reference] = [
            .video(H3VideoReference(frames: (0..<30).map { numberedFrame($0) }, fps: 24)),
            .audio(H3AudioReference(audio: silence(channels: 1, frameCount: 32_000, sampleRate: 32_000))),
            .image(H3ImageReference(image: image(512, 512))),
        ]
        let normalized = try H3ReferenceNormalizer.normalize(references, numFrames: 124)
        #expect(normalized.map(\.kind) == [.video, .audio, .image])
    }
}

@Suite struct ReferenceFrameRateTests {
    /// `slots[i] = floor(i * 24/fps + 0.5)`; frame `i` is held until the next frame's slot.
    @Test func thirtyToTwentyFour() {
        let indices = H3ReferenceNormalizer.resampleFrameIndices(
            sourceCount: 10, fps: 30, targetFPS: 24, limit: 100)
        // scale 0.8: slots = 0,1,2,2,3,4,5,6,6,7 ; end = floor(8.5) = 8. Frames 2 and 7 get a
        // zero-length hold and are dropped. Checked against
        // `np.repeat(arange(10), diff(slots, append=8))`.
        #expect(indices == [0, 1, 3, 4, 5, 6, 8, 9])
    }

    @Test func twentyFourToTwentyFourIsIdentity() {
        let indices = H3ReferenceNormalizer.resampleFrameIndices(
            sourceCount: 7, fps: 24, targetFPS: 24, limit: 100)
        #expect(indices == [0, 1, 2, 3, 4, 5, 6])
    }

    @Test func ntscTwentyThreeNineSevenSix() {
        // 24000/1001: slightly *fewer* source frames than output slots, so one frame duplicates.
        let fps = 24000.0 / 1001.0
        let indices = H3ReferenceNormalizer.resampleFrameIndices(
            sourceCount: 48, fps: fps, targetFPS: 24, limit: 200)
        #expect(indices.count == 48)
        // scale = 1.000999..., so slot 0..<499 track 1:1 and the first duplicate lands late.
        #expect(indices.prefix(10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        #expect(indices.last == 47)
    }

    @Test func twelveToTwentyFourDuplicatesEveryFrame() {
        let indices = H3ReferenceNormalizer.resampleFrameIndices(
            sourceCount: 5, fps: 12, targetFPS: 24, limit: 100)
        #expect(indices == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    }

    @Test func limitTruncates() {
        let indices = H3ReferenceNormalizer.resampleFrameIndices(
            sourceCount: 100, fps: 12, targetFPS: 24, limit: 5)
        #expect(indices == [0, 0, 1, 1, 2])
    }

    @Test func sourceFramesNeededCoversTheTruncation() {
        // Decoding this many source frames must reproduce the same truncated output as decoding all.
        for fps in [12.0, 25.0, 30.0, 60.0, 24000.0 / 1001.0] {
            let numFrames = 124
            let needed = H3ReferenceNormalizer.sourceFramesNeeded(numFrames: numFrames, fps: fps)
            let whole = H3ReferenceNormalizer.resampleFrameIndices(
                sourceCount: 4000, fps: fps, targetFPS: 24, limit: numFrames)
            let clipped = H3ReferenceNormalizer.resampleFrameIndices(
                sourceCount: needed, fps: fps, targetFPS: 24, limit: numFrames)
            #expect(whole == clipped, "fps \(fps): decoding \(needed) source frames must suffice")
        }
    }

    @Test func normalizedFramesTruncateAndKeepOrder() throws {
        // Frames already at the 16:9 canvas take the pass-through route, so this isolates the
        // 24 fps resample + truncation from the (separately covered, and slow) LANCZOS pass.
        // 12 fps doubles every frame: 124 output slots come from source frames 0...61.
        let frames = (0..<200).map { index in
            var frame = image(1344, 768, value: 3)
            frame.pixels[0] = UInt8(index % 256)
            return frame
        }
        let normalized = try H3ReferenceNormalizer.normalizeVideoFrames(frames, fps: 12, numFrames: 124)
        #expect(normalized.count == 124)
        #expect(normalized.map { Int($0.pixels[0]) } == (0..<62).flatMap { [$0, $0] })
    }

    @Test func alreadyOnCanvasPassesThroughUntouched() throws {
        // 1344x768 is exactly the 16:9 canvas: no resampling pass at all, bytes preserved.
        var frame = image(1344, 768)
        frame.pixels[17] = 200
        let normalized = try H3ReferenceNormalizer.normalizeVideoFrames([frame], fps: 24, numFrames: 124)
        #expect(normalized.count == 1)
        #expect(normalized[0].pixels == frame.pixels)
    }

    /// A clip too short to yield two 2 fps samples cannot be presented to the conditioner, and
    /// `normalize` says so before anything loads.
    @Test func rejectsAVideoTooShortToPresent() {
        let frames = (0..<10).map { _ in image(64, 64) }
        #expect(throws: H3Error.self) {
            try H3ReferenceNormalizer.normalize(
                [.video(H3VideoReference(frames: frames, fps: 24))], numFrames: 124)
        }
    }

    @Test func rejectsNonPositiveFrameRate() {
        #expect(throws: H3Error.self) {
            try H3ReferenceNormalizer.normalizeVideoFrames([image(64, 64)], fps: 0, numFrames: 124)
        }
    }
}

@Suite struct ReferenceImageTests {
    /// Geometry only: a 2048-edge LANCZOS pass costs tens of seconds in Debug, and the resample
    /// itself is covered bit-exactly by the `ref-normalize` parity probe.
    @Test func shortEdgeGoesTo2048WithNoAreaCap() throws {
        let size = try H3ReferenceNormalizer.referenceImageSize(width: 1920, height: 1280)
        #expect(size.width == 3072 && size.height == 2048)
        // Well past the 768*1344 canvas cap — that rule deliberately does not apply here.
        #expect(size.width * size.height > H3Constants.maxPixels)
    }

    @Test func upscalesSmallImages() throws {
        let size = try H3ReferenceNormalizer.referenceImageSize(width: 100, height: 100)
        #expect(size.width == 2048 && size.height == 2048)
    }

    @Test func roundsBothAxesToThirtyTwo() throws {
        // 1000x700 -> scale 2048/700; height 2048, width 2925.7 -> round(91.43)*32 = 2912.
        let size = try H3ReferenceNormalizer.referenceImageSize(width: 1000, height: 700)
        #expect(size.height == 2048)
        #expect(size.width == 2912)
        #expect(size.width % 32 == 0)
    }

    /// A 4:1 image at the short edge 2048 lands on exactly Qwen3-VL's image `max_pixels`
    /// (16 777 216), which is why `smart_resize` never rescales a normalized reference image.
    @Test func widestAcceptedImageExactlyFillsTheProcessorBudget() throws {
        let size = try H3ReferenceNormalizer.referenceImageSize(width: 4000, height: 1000)
        #expect(size.height == 2048 && size.width == 8192)
        #expect(size.height * size.width == 16_777_216)
    }

    @Test func aspectRatioOutsideOneToFourRejected() {
        #expect(throws: H3Error.self) {
            try H3ReferenceNormalizer.referenceImageSize(width: 100, height: 500)
        }
        #expect(throws: H3Error.self) {
            try H3ReferenceNormalizer.referenceImageSize(width: 500, height: 100)
        }
        // Exactly 1:4 is accepted.
        #expect(throws: Never.self) {
            try H3ReferenceNormalizer.referenceImageSize(width: 128, height: 512)
        }
    }

    /// One real resample, on the cheapest case that still exercises the path: an image already at
    /// its own reference size passes through untouched.
    @Test func alreadyAtReferenceSizePassesThroughUntouched() throws {
        var picture = image(2048, 2048, value: 60)
        picture.pixels[42] = 199
        let normalized = try H3ReferenceNormalizer.normalizeImage(picture)
        #expect(normalized.pixels == picture.pixels)
    }
}

@Suite struct ReferenceAudioTests {
    @Test func monoUpmixedByChannelRepeat() throws {
        let ramp = (0..<32_000).map { Float($0) / 32_000.0 }
        let buffer = H3AudioBuffer(channels: 1, frameCount: 32_000, samples: ramp, sampleRate: 32_000)
        let normalized = try H3ReferenceNormalizer.normalizeAudio(buffer, maxDuration: 5.0)
        #expect(normalized.channels == 2)
        #expect(normalized.frameCount == 32_000)
        #expect(Array(normalized.channel(0)) == Array(normalized.channel(1)))
    }

    @Test func truncationHappensAtTheNativeRate() throws {
        // 44.1 kHz, 3 s of material, truncated to 2.5 s: 110250 native samples, *then* resampled.
        // Resampling first would have kept ceil(320 * 132300/441) = 96000 and cut 80000 — the same
        // count here, but the boundary sample differs; the invariant under test is the count from
        // the native truncation, ceil(320 * 110250 / 441) = 80000.
        let samples = (0..<132_300).map { Float(sin(Double($0) * 0.01)) }
        let buffer = H3AudioBuffer(channels: 1, frameCount: 132_300, samples: samples, sampleRate: 44_100)
        let normalized = try H3ReferenceNormalizer.normalizeAudio(buffer, maxDuration: 2.5)
        #expect(normalized.sampleRate == 32_000)
        #expect(normalized.frameCount == 80_000)
    }

    @Test func alreadyAtTheVAERateIsNotResampled() throws {
        let samples = (0..<64_000).map { Float($0 % 17) / 17.0 }
        let buffer = H3AudioBuffer(channels: 2, frameCount: 32_000, samples: samples, sampleRate: 32_000)
        let normalized = try H3ReferenceNormalizer.normalizeAudio(buffer, maxDuration: 10.0)
        #expect(normalized.samples == samples)
    }

    @Test func nilSampleRateMeansTheVAERate() throws {
        let buffer = silence(channels: 2, frameCount: 32_000, sampleRate: nil)
        let normalized = try H3ReferenceNormalizer.normalizeAudio(buffer, maxDuration: 1.0)
        #expect(normalized.sampleRate == 32_000)
        #expect(normalized.frameCount == 32_000)
    }

    @Test func rejectsMoreThanStereo() {
        let buffer = silence(channels: 6, frameCount: 1000, sampleRate: 48_000)
        #expect(throws: H3Error.self) { try H3ReferenceNormalizer.normalizeAudio(buffer, maxDuration: 5) }
    }
}

@Suite struct ResamplerTests {
    /// Kernel geometry against torchaudio's `_get_sinc_resample_kernel` (GCD-reduced rates).
    @Test func kernelGeometryMatchesTorchaudio() {
        let cases: [(Int, Int, Int, Int, Int)] = [
            // origFreq, newFreq, expected kernels, expected width, expected taps
            (441, 320, 320, 9, 459),
            (3, 2, 2, 10, 23),
            (1, 2, 2, 7, 15),
        ]
        for (orig, new, kernelCount, width, taps) in cases {
            let (kernels, computedWidth) = H3Resampler.sincKernels(origFreq: orig, newFreq: new)
            #expect(kernels.count == kernelCount)
            #expect(computedWidth == width)
            #expect(kernels[0].count == taps)
        }
    }

    /// End-to-end against `torchaudio.transforms.Resample`, which is the transform the reference
    /// implementation used. Values captured from torchaudio 2.x, float32.
    @Test func matchesTorchaudio44100To32000() throws {
        let input = (0..<4410).map {
            Float(sin(2 * Double.pi * 440 * Double($0) / 44100) + 0.3 * sin(2 * Double.pi * 3000 * Double($0) / 44100))
        }
        let output = try H3Resampler.resample(input, from: 44_100, to: 32_000)
        #expect(output.count == 3200)
        let head: [Float] = [0.015003, 0.2498268, 0.450638, 0.5498185, 0.5514348, 0.4772675, 0.3809377, 0.3194339]
        let tail: [Float] = [-0.551435, -0.5498182, -0.450639, -0.2498279]
        for (index, expected) in head.enumerated() {
            #expect(abs(output[index] - expected) < 2e-5, "head[\(index)] \(output[index]) vs \(expected)")
        }
        for (index, expected) in tail.enumerated() {
            let position = output.count - tail.count + index
            #expect(abs(output[position] - expected) < 2e-5, "tail[\(index)] \(output[position]) vs \(expected)")
        }
    }

    @Test func matchesTorchaudio48000To32000() throws {
        let input = (0..<4800).map { Float(sin(2 * Double.pi * 440 * Double($0) / 48000)) }
        let output = try H3Resampler.resample(input, from: 48_000, to: 32_000)
        #expect(output.count == 3200)
        let head: [Float] = [0.0056462, 0.0853517, 0.1725593, 0.2562023, 0.3390411, 0.4188225]
        for (index, expected) in head.enumerated() {
            #expect(abs(output[index] - expected) < 2e-5, "head[\(index)] \(output[index]) vs \(expected)")
        }
    }

    @Test func identityWhenRatesMatch() throws {
        let input: [Float] = [1, 2, 3, 4]
        #expect(try H3Resampler.resample(input, from: 32_000, to: 32_000) == input)
    }
}

@Suite struct DisplayRotationTests {
    /// `av_display_rotation_get` is `-atan2(b, a)` over the ISO display matrix, whose `a`/`b` are a
    /// `CGAffineTransform`'s; the reference snaps it to `round(rotation / 90) % 4` clockwise turns.
    @Test func quarterTurnsFromPreferredTransform() {
        #expect(H3MediaDecoder.quarterTurns(fromPreferredTransform: .identity) == 0)
        // iPhone portrait: (a: 0, b: 1) -> rotation -90 -> turns 3.
        #expect(H3MediaDecoder.quarterTurns(
            fromPreferredTransform: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 0, ty: 0)) == 3)
        // rotation +90 -> turns 1.
        #expect(H3MediaDecoder.quarterTurns(
            fromPreferredTransform: CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 0)) == 1)
        // 180.
        #expect(H3MediaDecoder.quarterTurns(
            fromPreferredTransform: CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: 0, ty: 0)) == 2)
        // Off-axis angles snap to the nearest quarter turn.
        let radians = 88.0 * Double.pi / 180.0
        #expect(H3MediaDecoder.quarterTurns(
            fromPreferredTransform: CGAffineTransform(
                a: cos(radians), b: -sin(radians), c: sin(radians), d: cos(radians), tx: 0, ty: 0)) == 1)
    }

    /// `np.rot90(k: -turns)`: `out[i][j] = src[H-1-j][i]` for one clockwise turn.
    @Test func clockwiseRotationMatchesNumpy() {
        // 3x2 (W x H) with distinct values.
        let source = H3KeyframeImage(
            width: 3, height: 2,
            pixels: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        let once = source.rotatedClockwise(quarterTurns: 1)
        #expect(once.width == 2 && once.height == 3)
        // np.rot90(m, -1) of [[0,1,2],[3,4,5]] is [[3,0],[4,1],[5,2]].
        #expect(once.pixels.enumerated().filter { $0.offset % 3 == 0 }.map(\.element) == [3, 0, 4, 1, 5, 2])

        let thrice = source.rotatedClockwise(quarterTurns: 3)
        // np.rot90(m, 1) is [[2,5],[1,4],[0,3]].
        #expect(thrice.pixels.enumerated().filter { $0.offset % 3 == 0 }.map(\.element) == [2, 5, 1, 4, 0, 3])

        #expect(source.rotatedClockwise(quarterTurns: 4).pixels == source.pixels)
        #expect(source.rotatedClockwise(quarterTurns: 2).rotatedClockwise(quarterTurns: 2).pixels == source.pixels)
    }
}
