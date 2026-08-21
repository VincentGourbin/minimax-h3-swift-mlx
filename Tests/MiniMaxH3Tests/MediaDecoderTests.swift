// MediaDecoderTests.swift - the AVFoundation ingestion path, round-tripped through a real file
// Copyright 2026 Vincent Gourbin
//
// No checkpoint needed: these write an MP4 with the project's own exporter and read it back.
// The point is the contract the reference relies on and that no unit test of pure functions can
// establish — that `AVAssetReader` vends video frames in PRESENTATION order. H.264 reorders
// B-frames on the wire, and a decoder that handed them over in decode order would scramble every
// video reference silently, at the one place where "the order is the request".

import AVFoundation
import Foundation
import MLX
import Testing
@testable import MiniMaxH3

/// `(count, 64, 64, 3)` uint8 frames whose brightness ramps with the frame index, so the order
/// survives lossy compression: consecutive frames differ by 8 levels, far above H.264 noise.
/// `count` is bounded so the ramp stays inside `UInt8`.
private func rampFrames(_ count: Int) -> MLXArray {
    precondition(count <= 29, "the 8-level ramp only fits \(29) frames in a UInt8")
    var pixels = [UInt8]()
    pixels.reserveCapacity(count * 64 * 64 * 3)
    for index in 0..<count {
        pixels.append(contentsOf: [UInt8](repeating: UInt8(20 + index * 8), count: 64 * 64 * 3))
    }
    return MLXArray(pixels, [count, 64, 64, 3])
}

/// `(count, 64, 64, 3)` uniform frames, for the checks that only count frames.
private func flatFrames(_ count: Int) -> MLXArray {
    MLXArray([UInt8](repeating: 128, count: count * 64 * 64 * 3), [count, 64, 64, 3])
}

@Suite struct MediaDecoderRoundTripTests {
    @Test func decodedFramesComeBackInPresentationOrder() async throws {
        let count = 24
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-decoder-order-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        // A silent second of stereo, because the exporter always writes both tracks.
        let audio = MLXArray.zeros([2, 32_000]).asType(.float32)
        try await H3VideoExporter.export(
            frames: rampFrames(count), audio: audio, fps: 24, sampleRate: 32_000, to: url)

        let reference = try await H3MediaDecoder.decodeVideo(at: url)
        #expect(reference.frames.count == count)
        #expect(abs(reference.fps - 24) < 0.01)
        #expect(reference.frames.allSatisfy { $0.width == 64 && $0.height == 64 })

        // Strictly increasing brightness == presentation order preserved. Compression moves the
        // absolute levels a little, so the assertion is on the ordering, not on the values.
        let levels = reference.frames.map { Int($0.pixels[0]) }
        for index in 1..<levels.count {
            #expect(levels[index] > levels[index - 1],
                    "frame \(index) is not brighter than \(index - 1): \(levels)")
        }
        // And the ramp really did span the range it was written with.
        #expect(levels[levels.count - 1] - levels[0] > 8 * (count - 1) - 24)
    }

    @Test func soundtrackComesBackAtItsOwnRateAsStereo() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-decoder-audio-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        // One second of a 440 Hz tone on the left, 880 Hz on the right.
        let sampleCount = 32_000
        var samples = [Float](repeating: 0, count: 2 * sampleCount)
        for index in 0..<sampleCount {
            samples[index] = Float(sin(2 * Double.pi * 440 * Double(index) / 32_000)) * 0.5
            samples[sampleCount + index] = Float(sin(2 * Double.pi * 880 * Double(index) / 32_000)) * 0.5
        }
        try await H3VideoExporter.export(
            frames: flatFrames(24), audio: MLXArray(samples, [2, sampleCount]),
            fps: 24, sampleRate: 32_000, to: url)

        let reference = try await H3MediaDecoder.decodeVideo(at: url)
        let soundtrack = try #require(reference.audio)
        #expect(soundtrack.channels == 2)
        // AAC pads and the rate travels with the media, so check the rate and a plausible length
        // rather than an exact sample count.
        #expect(soundtrack.sampleRate == 32_000)
        #expect(soundtrack.frameCount > sampleCount / 2)
        // The two channels carry different content — a mono downmix would make them identical.
        let left = Array(soundtrack.channel(0).prefix(4000))
        let right = Array(soundtrack.channel(1).prefix(4000))
        #expect(left != right)
    }

    @Test func decodingBoundsTheSourceAtMaxDuration() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-decoder-bound-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        // 4 s at 24 fps; asking for 1 s must not decode all of it.
        try await H3VideoExporter.export(
            frames: flatFrames(96), audio: MLXArray.zeros([2, 128_000]).asType(.float32),
            fps: 24, sampleRate: 32_000, to: url)

        let whole = try await H3MediaDecoder.decodeVideo(at: url)
        #expect(whole.frames.count == 96)
        // One second plus the decoder's own slack — well short of the full clip.
        let bounded = try await H3MediaDecoder.decodeVideo(at: url, maxDuration: 1.0)
        #expect(bounded.frames.count < 96)
        #expect(bounded.frames.count >= 24)
    }
}
