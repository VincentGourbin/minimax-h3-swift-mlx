// Ref2VAPacking.swift - the `[text | reference blocks | target audio | target video]` layout
// Copyright 2026 Vincent Gourbin
//
// Port of `MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`.
//
// The reference order advances the shared audio/video rotary clock, so it is part of the LAYOUT
// rather than a detail of the presentation. Each block pushes the clock forward by the time it
// occupies:
//   image  -> exactly +1.0     (one integer slot, NOT a latent frame's 5/3 units)
//   audio  -> +latentCount     (1.0 per audio latent, the 40 Hz grid)
//   video  -> +max(audioLatents, videoSpan)
// and the generated rows then start from wherever the blocks left it.
//
// A video reference's soundtrack rows are packed IMMEDIATELY BEFORE its video rows and share their
// rotary origin: that alignment is the AV sync of the reference, exactly as it is for the target.

import Foundation
import MLX

/// The latent geometry of one reference block, as the encoders produced it. The layout is built
/// from these shapes rather than from the media, so the two can never disagree.
public struct H3ReferenceBlockGeometry: Sendable {
    public var kind: H3ReferenceKind
    /// Video side, for an image or video reference: the shape of its conditioning latents.
    /// An image encodes to a single latent frame.
    public var latentFrames: Int
    public var latentHeight: Int
    public var latentWidth: Int
    /// Audio side: the rows its soundtrack packs into (`latents * 2`, channel-major). 0 when the
    /// reference carries no waveform.
    public var audioRows: Int

    public init(
        kind: H3ReferenceKind, latentFrames: Int = 0, latentHeight: Int = 0, latentWidth: Int = 0,
        audioRows: Int = 0
    ) {
        self.kind = kind
        self.latentFrames = latentFrames
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.audioRows = audioRows
    }
}

extension H3Packing {
    /// The `(h, w)` rotary coordinates of one latent frame, h-major, plus the width axis they were
    /// built from (audio rows pin to its two extremes).
    static func framePositionGrid(
        latentHeight: Int, latentWidth: Int, patchH: Int, patchW: Int
    ) -> (grid: [(Double, Double)], widthGrid: [Double]) {
        let sqrtArea = (Double(latentHeight) * Double(latentWidth)).squareRoot()
        let heightGrid = spatialPositionGrid(dim: latentHeight, patch: patchH, sqrtArea: sqrtArea)
        let widthGrid = spatialPositionGrid(dim: latentWidth, patch: patchW, sqrtArea: sqrtArea)
        var grid = [(Double, Double)]()
        grid.reserveCapacity(heightGrid.count * widthGrid.count)
        for h in heightGrid { for w in widthGrid { grid.append((h, w)) } }
        return (grid, widthGrid)
    }

    /// Place one channel-major audio block: no height coordinate, width pinned to the two extremes
    /// of the width grid of *its own* block — the target grid for a standalone audio reference, the
    /// video's own grid for a soundtrack.
    private static func fillAudioPositions(
        _ positions: inout [Double], start: Int, numAudioLatents: Int, rotaryTime: Double,
        widthGrid: [Double]
    ) {
        for channel in 0..<H3Constants.audioChannels {
            let pinnedWidth = channel == 0 ? widthGrid[0] : widthGrid[widthGrid.count - 1]
            for latent in 0..<numAudioLatents {
                let base = (start + channel * numAudioLatents + latent) * 3
                positions[base] = rotaryTime + Double(latent)
                positions[base + 2] = pinnedWidth
            }
        }
    }

    /// Build the `[text | reference blocks | target audio | target video]` layout of a `ref2va`
    /// request.
    ///
    /// - Parameters:
    ///   - textTokenTags: modality tag of every text row (text 1, a reference's vision block 0).
    ///   - blocks: one entry per reference, **in request order**.
    public static func buildRef2VAPackedSequence(
        textTokenTags: [Int32],
        blocks: [H3ReferenceBlockGeometry],
        numLatentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        numAudioLatents: Int,
        patchSize: (t: Int, h: Int, w: Int)
    ) throws -> H3PackedSequence {
        let numTextTokens = textTokenTags.count
        let targetRowsPerFrame = (latentHeight / patchSize.h) * (latentWidth / patchSize.w)
        let numTargetVideoRows = numLatentFrames * targetRowsPerFrame
        let numTargetAudioRows = numAudioLatents * H3Constants.audioChannels

        func videoRows(_ block: H3ReferenceBlockGeometry) -> Int {
            block.latentFrames * (block.latentHeight / patchSize.h) * (block.latentWidth / patchSize.w)
        }
        let numReferenceVideoRows = blocks
            .filter { $0.kind != .audio }
            .reduce(0) { $0 + videoRows($1) }
        let numReferenceAudioRows = blocks.reduce(0) { $0 + $1.audioRows }
        let sequenceLength = numTextTokens + numReferenceVideoRows + numReferenceAudioRows
            + numTargetAudioRows + numTargetVideoRows

        var positions = [Double](repeating: 0, count: sequenceLength * 3)
        for index in 0..<numTextTokens { positions[index * 3] = Double(index) }

        let (targetFrameGrid, targetWidthGrid) = framePositionGrid(
            latentHeight: latentHeight, latentWidth: latentWidth,
            patchH: patchSize.h, patchW: patchSize.w)

        var videoIndexValues = [Int32]()
        var audioIndexValues = [Int32]()
        videoIndexValues.reserveCapacity(numReferenceVideoRows + numTargetVideoRows)
        audioIndexValues.reserveCapacity(numReferenceAudioRows + numTargetAudioRows)

        var cursor = numTextTokens
        var rotaryTime = Double(numTextTokens)

        for block in blocks {
            switch block.kind {
            case .image:
                let rows = videoRows(block)
                let (grid, _) = framePositionGrid(
                    latentHeight: block.latentHeight, latentWidth: block.latentWidth,
                    patchH: patchSize.h, patchW: patchSize.w)
                // An image encodes to exactly one latent frame — the spatial encoder alone, with
                // no temporal extent to chunk — and takes one integer rotary slot for it. More
                // than one would mean the encoder ran the temporal path over a repeated frame.
                guard block.latentFrames == 1, grid.count == rows else {
                    throw H3Error.invalidInput(
                        "An image reference must encode to a single latent frame, got "
                            + "\(block.latentFrames).")
                }
                for (offset, (h, w)) in grid.enumerated() {
                    let base = (cursor + offset) * 3
                    positions[base] = rotaryTime
                    positions[base + 1] = h
                    positions[base + 2] = w
                }
                videoIndexValues.append(contentsOf: (cursor..<(cursor + rows)).map(Int32.init))
                cursor += rows
                // An image is a single frame and takes a single integer rotary slot.
                rotaryTime += 1.0

            case .audio:
                let rows = block.audioRows
                let latents = rows / H3Constants.audioChannels
                fillAudioPositions(
                    &positions, start: cursor, numAudioLatents: latents, rotaryTime: rotaryTime,
                    widthGrid: targetWidthGrid)
                audioIndexValues.append(contentsOf: (cursor..<(cursor + rows)).map(Int32.init))
                cursor += rows
                rotaryTime += Double(latents)

            case .video:
                let audioRows = block.audioRows
                let audioLatents = audioRows / H3Constants.audioChannels
                let rows = videoRows(block)
                let (grid, widthGrid) = framePositionGrid(
                    latentHeight: block.latentHeight, latentWidth: block.latentWidth,
                    patchH: patchSize.h, patchW: patchSize.w)

                // Soundtrack first, sharing the video's rotary origin.
                if audioRows > 0 {
                    fillAudioPositions(
                        &positions, start: cursor, numAudioLatents: audioLatents,
                        rotaryTime: rotaryTime, widthGrid: widthGrid)
                    audioIndexValues.append(
                        contentsOf: (cursor..<(cursor + audioRows)).map(Int32.init))
                    cursor += audioRows
                }

                let frameTimes = temporalPositionGrid(
                    numLatentFrames: block.latentFrames, origin: rotaryTime)
                for frame in 0..<block.latentFrames {
                    for (offset, (h, w)) in grid.enumerated() {
                        let base = (cursor + frame * grid.count + offset) * 3
                        positions[base] = frameTimes[frame]
                        positions[base + 1] = h
                        positions[base + 2] = w
                    }
                }
                videoIndexValues.append(contentsOf: (cursor..<(cursor + rows)).map(Int32.init))
                cursor += rows
                rotaryTime += max(
                    Double(audioLatents),
                    temporalPositionSpan(numLatentFrames: block.latentFrames))
            }
        }

        // The generated rows share the origin the reference blocks left behind.
        let audioStart = cursor
        let videoStart = audioStart + numTargetAudioRows
        fillAudioPositions(
            &positions, start: audioStart, numAudioLatents: numAudioLatents,
            rotaryTime: rotaryTime, widthGrid: targetWidthGrid)
        let targetFrameTimes = temporalPositionGrid(
            numLatentFrames: numLatentFrames, origin: rotaryTime)
        for frame in 0..<numLatentFrames {
            for (offset, (h, w)) in targetFrameGrid.enumerated() {
                let base = (videoStart + frame * targetRowsPerFrame + offset) * 3
                positions[base] = targetFrameTimes[frame]
                positions[base + 1] = h
                positions[base + 2] = w
            }
        }
        audioIndexValues.append(contentsOf: (audioStart..<videoStart).map(Int32.init))
        videoIndexValues.append(contentsOf: (videoStart..<sequenceLength).map(Int32.init))

        // Tags: text rows carry the presentation's own tagging (vision blocks already 0), every
        // audio row the audio tag, every video row — reference and target alike — the video tag.
        var tagValues = [Int32](repeating: H3Constants.textTag, count: sequenceLength)
        for (index, tag) in textTokenTags.enumerated() { tagValues[index] = tag }
        for index in audioIndexValues { tagValues[Int(index)] = H3Constants.audioTag }
        for index in videoIndexValues { tagValues[Int(index)] = H3Constants.videoTag }

        // Where every row of `[text | video rows | audio rows]` lands in the sequence. The
        // transformer concatenates its three streams in that order and gathers with this, which is
        // what reproduces the reference's `index_copy` scatter over an interleaved layout.
        var gatherOrder = [Int32](repeating: 0, count: sequenceLength)
        for (row, position) in ((0..<numTextTokens).map(Int32.init)
            + videoIndexValues + audioIndexValues).enumerated() {
            gatherOrder[Int(position)] = Int32(row)
        }

        let positionsFloat = positions.map(Float.init)
        return H3PackedSequence(
            sequenceLength: sequenceLength,
            positionIds: MLXArray(positionsFloat, [sequenceLength, 3]),
            tokenTags: MLXArray(tagValues),
            videoIndices: MLXArray(videoIndexValues),
            audioIndices: MLXArray(audioIndexValues),
            textIndices: MLXArray((0..<numTextTokens).map(Int32.init)),
            numConditionVideoRows: numReferenceVideoRows,
            numConditionAudioRows: numReferenceAudioRows,
            tokenTagValues: tagValues,
            videoIndexValues: videoIndexValues,
            audioIndexValues: audioIndexValues,
            packedGatherOrder: MLXArray(gatherOrder),
            numTextTokens: numTextTokens,
            numAudioRows: numReferenceAudioRows + numTargetAudioRows,
            numVideoRows: numReferenceVideoRows + numTargetVideoRows
        )
    }
}
