// Ref2VAPackingTests.swift - ref2va Phase B geometry, no checkpoint required
// Copyright 2026 Vincent Gourbin
//
// Contracts under test come from `build_ref2va_packed_sequence`, `build_row_timesteps` and
// `_sample_video_condition_frames`.

import MLX
import Testing
@testable import MiniMaxH3

private let patch = (t: 1, h: 2, w: 2)

/// Target geometry of the canonical 5 s 16:9 request, shrunk so the tests stay cheap:
/// 4 latent frames on a 8x16 latent canvas (4x8 = 32 rows per frame), 10 audio latents.
private let targetFrames = 4
private let targetH = 8
private let targetW = 16
private let targetRowsPerFrame = (targetH / patch.h) * (targetW / patch.w)  // 32
private let targetAudioLatents = 10

private func layout(
    textTokens: Int, blocks: [H3ReferenceBlockGeometry]
) throws -> H3PackedSequence {
    try H3Packing.buildRef2VAPackedSequence(
        textTokenTags: [Int32](repeating: H3Constants.textTag, count: textTokens),
        blocks: blocks,
        numLatentFrames: targetFrames,
        latentHeight: targetH,
        latentWidth: targetW,
        numAudioLatents: targetAudioLatents,
        patchSize: patch)
}

private func positions(_ sequence: H3PackedSequence) -> [[Float]] {
    let flat = sequence.positionIds.asArray(Float.self)
    return (0..<sequence.sequenceLength).map { Array(flat[($0 * 3)..<($0 * 3 + 3)]) }
}

@Suite struct Ref2VALayoutTests {
    @Test func imageReferenceAdvancesTheClockByExactlyOne() throws {
        let text = 5
        let image = H3ReferenceBlockGeometry(
            kind: .image, latentFrames: 1, latentHeight: 4, latentWidth: 4)
        let sequence = try layout(textTokens: text, blocks: [image])
        let imageRows = 1 * (4 / patch.h) * (4 / patch.w)  // 4
        #expect(sequence.numConditionVideoRows == imageRows)
        #expect(sequence.numConditionAudioRows == 0)

        let p = positions(sequence)
        // Every row of the image block sits on one rotary instant: where the text ended.
        for row in text..<(text + imageRows) { #expect(p[row][0] == Float(text)) }
        // The generated rows start one integer slot later — NOT a latent frame's 5/3.
        let audioStart = text + imageRows
        #expect(p[audioStart][0] == Float(text) + 1.0)
        let videoStart = audioStart + targetAudioLatents * 2
        #expect(p[videoStart][0] == Float(text) + 1.0)
    }

    @Test func audioReferenceAdvancesByItsLatentCount() throws {
        let text = 3
        let latents = 7
        let audio = H3ReferenceBlockGeometry(kind: .audio, audioRows: latents * 2)
        let picture = H3ReferenceBlockGeometry(
            kind: .image, latentFrames: 1, latentHeight: 4, latentWidth: 4)
        let sequence = try layout(textTokens: text, blocks: [audio, picture])

        #expect(sequence.numConditionAudioRows == latents * 2)
        let p = positions(sequence)
        // Audio rows are channel-major: latent l of channel c at rotaryTime + l.
        for channel in 0..<2 {
            for latent in 0..<latents {
                #expect(p[text + channel * latents + latent][0] == Float(text + latent))
            }
        }
        // The image block that follows starts `latents` later.
        #expect(p[text + latents * 2][0] == Float(text + latents))
    }

    @Test func videoReferencePacksItsSoundtrackImmediatelyBefore() throws {
        let text = 2
        let audioLatents = 3
        let video = H3ReferenceBlockGeometry(
            kind: .video, latentFrames: 2, latentHeight: 4, latentWidth: 4,
            audioRows: audioLatents * 2)
        let sequence = try layout(textTokens: text, blocks: [video])

        let videoRows = 2 * (4 / patch.h) * (4 / patch.w)  // 8
        #expect(sequence.numConditionVideoRows == videoRows)
        #expect(sequence.numConditionAudioRows == audioLatents * 2)

        // The soundtrack occupies the rows right after the text, the video's rows right after it.
        let audioBlock = (text..<(text + audioLatents * 2)).map(Int32.init)
        #expect(Array(sequence.audioIndexValues.prefix(audioLatents * 2)) == audioBlock)
        let videoBlockStart = text + audioLatents * 2
        #expect(sequence.videoIndexValues[0] == Int32(videoBlockStart))

        // Both share the same rotary origin — that alignment IS the reference's AV sync.
        let p = positions(sequence)
        #expect(p[text][0] == Float(text))
        #expect(p[videoBlockStart][0] == Float(text))

        // The clock advances by max(audioLatents, videoSpan). Two latent frames span
        // 5/3 * (1 + 4) = 8.333..., which wins over 3 audio latents.
        let span = Float(5.0 / 3.0 * (1.0 + 4.0))
        let audioStart = videoBlockStart + videoRows
        #expect(abs(p[audioStart][0] - (Float(text) + span)) < 1e-4)
    }

    @Test func aLongSoundtrackWinsTheMaxOverAShortVideo() throws {
        let text = 1
        // One latent frame spans 5/3; 9 audio latents are longer, so the audio side sets the clock.
        let video = H3ReferenceBlockGeometry(
            kind: .video, latentFrames: 1, latentHeight: 4, latentWidth: 4, audioRows: 9 * 2)
        let sequence = try layout(textTokens: text, blocks: [video])
        let p = positions(sequence)
        let generatedAudioStart = text + 9 * 2 + 1 * 4
        #expect(abs(p[generatedAudioStart][0] - Float(text + 9)) < 1e-4)
    }

    @Test func referenceOrderIsTheLayout() throws {
        // Same three references, two orders -> two different rotary layouts.
        let picture = H3ReferenceBlockGeometry(
            kind: .image, latentFrames: 1, latentHeight: 4, latentWidth: 4)
        let sound = H3ReferenceBlockGeometry(kind: .audio, audioRows: 6 * 2)
        let first = try layout(textTokens: 2, blocks: [picture, sound])
        let second = try layout(textTokens: 2, blocks: [sound, picture])
        #expect(first.sequenceLength == second.sequenceLength)
        #expect(positions(first) != positions(second))
    }

    @Test func tagsFollowModalityNotPosition() throws {
        let text = 4
        let video = H3ReferenceBlockGeometry(
            kind: .video, latentFrames: 2, latentHeight: 4, latentWidth: 4, audioRows: 3 * 2)
        let sequence = try layout(textTokens: text, blocks: [video])
        let tags = sequence.tokenTagValues
        #expect(tags.prefix(text).allSatisfy { $0 == H3Constants.textTag })
        for index in sequence.audioIndexValues { #expect(tags[Int(index)] == H3Constants.audioTag) }
        for index in sequence.videoIndexValues { #expect(tags[Int(index)] == H3Constants.videoTag) }
        #expect(tags.count == sequence.sequenceLength)
    }

    @Test func rowCountsAddUp() throws {
        let picture = H3ReferenceBlockGeometry(
            kind: .image, latentFrames: 1, latentHeight: 4, latentWidth: 8)
        let video = H3ReferenceBlockGeometry(
            kind: .video, latentFrames: 3, latentHeight: 8, latentWidth: 8, audioRows: 5 * 2)
        let sound = H3ReferenceBlockGeometry(kind: .audio, audioRows: 2 * 2)
        let sequence = try layout(textTokens: 6, blocks: [picture, video, sound])

        let pictureRows = 1 * 2 * 4
        let videoRows = 3 * 4 * 4
        let expected = 6 + pictureRows + videoRows + (5 * 2) + (2 * 2)
            + targetAudioLatents * 2 + targetFrames * targetRowsPerFrame
        #expect(sequence.sequenceLength == expected)
        #expect(sequence.numConditionVideoRows == pictureRows + videoRows)
        #expect(sequence.numConditionAudioRows == 5 * 2 + 2 * 2)
        #expect(sequence.videoIndexValues.count == pictureRows + videoRows + targetFrames * targetRowsPerFrame)
        #expect(sequence.audioIndexValues.count == 5 * 2 + 2 * 2 + targetAudioLatents * 2)
        // Every row is claimed exactly once.
        let claimed = Set(sequence.videoIndexValues) .union(sequence.audioIndexValues)
            .union((0..<6).map(Int32.init))
        #expect(claimed.count == sequence.sequenceLength)
    }

    @Test func generatedRowsInheritTheClockTheBlocksLeft() throws {
        let text = 3
        let blocks = [
            H3ReferenceBlockGeometry(kind: .image, latentFrames: 1, latentHeight: 4, latentWidth: 4),
            H3ReferenceBlockGeometry(kind: .audio, audioRows: 4 * 2),
        ]
        let sequence = try layout(textTokens: text, blocks: blocks)
        let p = positions(sequence)
        // 3 (text) + 1 (image) + 4 (audio latents) = 8.
        let origin = Float(text) + 1.0 + 4.0
        let audioStart = Int(sequence.audioIndexValues[4 * 2])
        let videoStart = Int(sequence.videoIndexValues[sequence.numConditionVideoRows])
        #expect(p[audioStart][0] == origin)
        #expect(p[videoStart][0] == origin)
    }

    /// The conditioning rows are a prefix of the GATHER ORDER, never of the sequence.
    @Test func rowTimestepsPinTheReferenceRows() throws {
        let text = 3
        let video = H3ReferenceBlockGeometry(
            kind: .video, latentFrames: 2, latentHeight: 4, latentWidth: 4, audioRows: 3 * 2)
        let sequence = try layout(textTokens: text, blocks: [video])
        let (timesteps, indices) = H3Packing.buildRowTimesteps(
            layout: sequence, videoTimestep: 0.4, audioTimestep: 0.6,
            conditionVideoTimestep: 0.999, conditionAudioTimestep: 1.0)
        let perRow = indices.asArray(Int32.self).map { timesteps[Int($0)] }

        for index in sequence.videoIndexValues.prefix(sequence.numConditionVideoRows) {
            #expect(perRow[Int(index)] == 0.999)
        }
        for index in sequence.audioIndexValues.prefix(sequence.numConditionAudioRows) {
            #expect(perRow[Int(index)] == 1.0)  // a soundtrack conditions clean, never noised
        }
        for index in sequence.audioIndexValues.dropFirst(sequence.numConditionAudioRows) {
            #expect(perRow[Int(index)] == 0.6)
        }
        for index in sequence.videoIndexValues.dropFirst(sequence.numConditionVideoRows) {
            #expect(perRow[Int(index)] == 0.4)
        }
        // Text rows never reach an output head and ride the video timestep.
        for row in 0..<text { #expect(perRow[row] == 0.4) }
    }
}

@Suite struct Ref2VAGatherOrderTests {
    /// The transformer concatenates `[text | video rows | audio rows]` and gathers with
    /// `packedGatherOrder` to reproduce the reference's `index_copy` scatter. This checks the
    /// gather actually inverts the three index lists — a wrong permutation would silently feed the
    /// blocks a scrambled sequence.
    @Test func gatherOrderInvertsTheIndexLists() throws {
        let blocks = [
            H3ReferenceBlockGeometry(kind: .audio, audioRows: 3 * 2),
            H3ReferenceBlockGeometry(
                kind: .video, latentFrames: 2, latentHeight: 4, latentWidth: 4, audioRows: 2 * 2),
            H3ReferenceBlockGeometry(kind: .image, latentFrames: 1, latentHeight: 4, latentWidth: 8),
        ]
        let text = 5
        let sequence = try layout(textTokens: text, blocks: blocks)
        let order = try #require(sequence.packedGatherOrder)

        // Mark every row of the three streams with its own identity, concatenate in the order the
        // transformer does, gather, and check each sequence position received the right row.
        let textRows = (0..<text).map { Float(1_000_000 + $0) }
        let videoRows = (0..<sequence.videoIndexValues.count).map { Float(2_000_000 + $0) }
        let audioRows = (0..<sequence.audioIndexValues.count).map { Float(3_000_000 + $0) }
        let concatenatedRows = MLXArray(textRows + videoRows + audioRows).reshaped(-1, 1)
        let gathered = concatenatedRows[order].reshaped(-1).asArray(Float.self)

        #expect(gathered.count == sequence.sequenceLength)
        for row in 0..<text { #expect(gathered[row] == Float(1_000_000 + row)) }
        for (index, position) in sequence.videoIndexValues.enumerated() {
            #expect(gathered[Int(position)] == Float(2_000_000 + index))
        }
        for (index, position) in sequence.audioIndexValues.enumerated() {
            #expect(gathered[Int(position)] == Float(3_000_000 + index))
        }
        // Every position written exactly once.
        #expect(Set(order.asArray(Int32.self)).count == sequence.sequenceLength)
    }

    /// A t2va / fl2va layout stays contiguous and carries no gather — the transformer's fast path.
    @Test func contiguousLayoutsCarryNoGatherOrder() throws {
        let sequence = try H3Packing.buildPackedSequence(
            textTokenTags: [Int32](repeating: H3Constants.textTag, count: 4),
            numLatentFrames: targetFrames, latentHeight: targetH, latentWidth: targetW,
            numAudioLatents: targetAudioLatents, patchSize: patch, keyframeAnchors: ["first"])
        #expect(sequence.packedGatherOrder == nil)
    }
}

@Suite struct Ref2VAConditionerSamplingTests {
    /// Values from the reference `_sample_video_condition_frames` at 24 -> 2 fps, patch 2.
    @Test func sampledFramesAndBlockLabels() throws {
        let cases: [(frames: Int, sampled: Int, labels: [String])] = [
            (124, 11, ["<0.2 seconds>", "<1.2 seconds>", "<2.2 seconds>", "<3.2 seconds>",
                       "<4.2 seconds>", "<5.0 seconds>"]),
            (122, 11, ["<0.2 seconds>", "<1.2 seconds>", "<2.2 seconds>", "<3.2 seconds>",
                       "<4.2 seconds>", "<5.0 seconds>"]),
            (24, 2, ["<0.2 seconds>"]),
            (25, 3, ["<0.2 seconds>", "<1.0 seconds>"]),
        ]
        for (frameCount, sampled, labels) in cases {
            let (indices, timestamps) = try H3ReferenceVideoProcessor.sampleConditionFrames(
                frameCount: frameCount)
            #expect(indices.count == sampled, "\(frameCount) frames")
            #expect(timestamps.map(H3ReferenceVideoProcessor.timestampLabel) == labels,
                    "\(frameCount) frames")
            #expect(indices == Swift.stride(from: 0, to: frameCount, by: 12).map { $0 })
        }
    }

    @Test func fifteenSecondReferenceRunsToFifteenPointZero() throws {
        let (indices, timestamps) = try H3ReferenceVideoProcessor.sampleConditionFrames(
            frameCount: 362)
        #expect(indices.count == 31)
        #expect(timestamps.count == 16)
        #expect(H3ReferenceVideoProcessor.timestampLabel(timestamps.last!) == "<15.0 seconds>")
    }

    @Test func tooShortToMergeIsRejected() {
        // 12 frames at 24 fps sample to a single frame, and a group needs two.
        #expect(throws: H3Error.self) {
            try H3ReferenceVideoProcessor.sampleConditionFrames(frameCount: 12)
        }
    }

    /// `smart_resize` is a pass-through for a normalized reference until the clip is long enough
    /// that `t * h * w` breaks Qwen3-VL's 25 165 824-pixel budget.
    @Test func smartResizeOnlyBitesOnLongReferences() throws {
        // 11 sampled frames (5 s) at the biggest canvas: t_bar 12, 12 * 1032192 fits.
        let short = try H3ReferenceVideoProcessor.smartResize(
            numFrames: 11, height: 768, width: 1344)
        #expect(short.height == 768 && short.width == 1344)
        // 31 sampled frames (15 s) at the same canvas: t_bar 32, 32 * 1032192 = 33 030 144 > budget.
        let long = try H3ReferenceVideoProcessor.smartResize(
            numFrames: 31, height: 768, width: 1344)
        #expect(long.height < 768 && long.width < 1344)
        #expect(long.height % 32 == 0 && long.width % 32 == 0)
    }
}
