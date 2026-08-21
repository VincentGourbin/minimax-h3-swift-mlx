// Packing.swift - Packed-sequence geometry of MiniMax-H3 (port of diffusers packing.py)
// Copyright 2026 Vincent Gourbin
//
// MiniMax-H3 runs its transformer over a single packed 1-D sequence holding every modality:
//   [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
// Everything here places a row in that sequence and gives it its (t, h, w) rotary coordinate.
// Coordinates are built in Double because video and audio share one 40-units-per-second rotary
// clock (video: 5/3 units per pixel frame at 24 fps, audio: 1 unit per latent at 40 latents/s)
// and that shared clock IS the audio/video alignment.

import Foundation
import MLX

// MARK: - Checkpoint constants

public enum H3Constants {
    // Per-row modality tags; they index the transformer's AdaLN table (checkpoint contract).
    public static let videoTag: Int32 = 0
    public static let textTag: Int32 = 1
    public static let audioTag: Int32 = 2

    public static let fps = 24
    public static let shortEdge = 768
    public static let maxPixels = 768 * 1344
    public static let canvasMultiple = 32
    public static let minAspectRatio = 0.25
    public static let maxAspectRatio = 4.0
    public static let minDuration = 5.0
    public static let maxDuration = 15.0

    // The video VAE encodes 17 pixel frames per chunk into 5 latent frames (drops 3 trailing).
    public static let framesPerChunk = 17
    public static let latentsPerChunk = 5

    // Pixel convention of the video VAE: ImageNet-normalized RGB over a [0, 1] base range.
    public static let pixelMean: [Float] = [0.485, 0.456, 0.406]
    public static let pixelStd: [Float] = [0.229, 0.224, 0.225]

    // Conditioning is the *unnormalized* hidden state after this Qwen3-VL decoder layer.
    public static let textEncoderLayer = 50

    // Audio VAE: 32 kHz, hop 800 -> 40 latents/s; stereo carried as two channel-major row blocks.
    public static let audioLatentsPerSecond = 40
    public static let audioChannels = 2
    public static let audioSampleRate = 32_000

    // Keyframe conditioning rows are noised to t = 0.999 and pinned there every step.
    public static let keyframeNoiseAug: Float = 0.999
    // The keyframe VAE posterior is sampled under this fixed seed, independent of the request seed.
    public static let keyframeEncodeSeed: UInt64 = 42

    // Rotary-time constants: one latent frame spans 5/3 * framesPerLatent units, pattern cyclic.
    static let ropeFrameRescale = 5.0 / 3.0
    static let ropeFramesPerLatent: [Double] = [1, 4, 4, 4, 4]
    static let ropeSpatialScale = 32.0
}

// MARK: - Frame / canvas arithmetic

public enum H3Geometry {
    /// Resolve a display aspect ratio into an H3 canvas: short edge 768, area capped at 768*1344,
    /// both axes rounded to the nearest multiple of 32. Returns (height, width).
    public static func resolveCanvasSize(aspectWidth: Double, aspectHeight: Double) throws -> (height: Int, width: Int) {
        guard aspectWidth > 0, aspectHeight > 0 else {
            throw H3Error.invalidInput("The aspect ratio must be positive, got \(aspectWidth):\(aspectHeight).")
        }
        let ratio = aspectWidth / aspectHeight
        guard ratio >= H3Constants.minAspectRatio, ratio <= H3Constants.maxAspectRatio else {
            throw H3Error.invalidInput("MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got \(ratio).")
        }

        var width: Double
        var height: Double
        if ratio >= 1.0 {
            width = Double(H3Constants.shortEdge) * ratio
            height = Double(H3Constants.shortEdge)
        } else {
            width = Double(H3Constants.shortEdge)
            height = Double(H3Constants.shortEdge) / ratio
        }
        let area = width * height
        if area > Double(H3Constants.maxPixels) {
            let scale = (Double(H3Constants.maxPixels) / area).squareRoot()
            width *= scale
            height *= scale
        }
        // Python `round` is half-to-even; a video reference's own aspect ratio lands on exact
        // halves often enough that half-away-from-zero would resolve a different canvas.
        let m = H3Constants.canvasMultiple
        let roundedHeight = max(m, Int((height / Double(m)).rounded(.toNearestOrEven)) * m)
        let roundedWidth = max(m, Int((width / Double(m)).rounded(.toNearestOrEven)) * m)
        return (roundedHeight, roundedWidth)
    }

    /// Snap a frame count up to the next `17 * n + 5` the video VAE can decode.
    public static func alignNumFrames(_ numFrames: Int) throws -> Int {
        guard numFrames >= 1 else { throw H3Error.invalidInput("`numFrames` must be positive, got \(numFrames).") }
        var frames = numFrames
        while frames % H3Constants.framesPerChunk != H3Constants.latentsPerChunk { frames += 1 }
        return frames
    }

    /// Latent frames for an aligned `17 * n + 5` frame count: `5 * n + 2`.
    public static func videoLatentNumFrames(_ numFrames: Int) throws -> Int {
        guard numFrames % H3Constants.framesPerChunk == H3Constants.latentsPerChunk else {
            throw H3Error.invalidInput("`numFrames` must be of the form 17 * n + 5, got \(numFrames).")
        }
        return (numFrames - H3Constants.latentsPerChunk) / H3Constants.framesPerChunk
            * H3Constants.latentsPerChunk + 2
    }

    /// Audio latents per channel covering `numFrames` frames at 24 fps, on the 40 Hz latent grid.
    public static func audioLatentNumFrames(_ numFrames: Int) -> Int {
        Int((Double(numFrames) / Double(H3Constants.fps) * Double(H3Constants.audioLatentsPerSecond)).rounded())
    }
}

// MARK: - Patchify / unpatchify

public enum H3Packing {
    /// Pack video latents `(1, C, F, H, W)` into transformer rows
    /// `(F/pt * H/ph * W/pw, C * pt * ph * pw)`, frame-major then row-major.
    public static func patchifyVideoLatents(_ latents: MLXArray, patchSize: (t: Int, h: Int, w: Int)) -> MLXArray {
        let (batch, channels, frames, height, width) =
            (latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3), latents.dim(4))
        precondition(
            frames % patchSize.t == 0 && height % patchSize.h == 0 && width % patchSize.w == 0,
            "Latents \(latents.shape) not divisible by patch \(patchSize)"
        )
        var x = latents.reshaped(
            batch, channels,
            frames / patchSize.t, patchSize.t,
            height / patchSize.h, patchSize.h,
            width / patchSize.w, patchSize.w
        )
        x = x.transposed(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshaped(-1, channels * patchSize.t * patchSize.h * patchSize.w)
    }

    /// Inverse of `patchifyVideoLatents`: rows back into `(1, C, F, H, W)` latents.
    public static func unpatchifyVideoTokens(
        _ rows: MLXArray,
        numLatentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        channels: Int,
        patchSize: (t: Int, h: Int, w: Int)
    ) -> MLXArray {
        var x = rows.reshaped(
            -1,
            numLatentFrames / patchSize.t,
            latentHeight / patchSize.h,
            latentWidth / patchSize.w,
            channels, patchSize.t, patchSize.h, patchSize.w
        )
        x = x.transposed(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshaped(-1, channels, numLatentFrames, latentHeight, latentWidth)
    }

    /// Unpack channel-major audio rows `(2 * A, C)` into `(2, C, A)` — one batch item per stereo
    /// channel, which is what the mono audio VAE consumes.
    public static func unpackAudioTokens(_ rows: MLXArray, numAudioLatents: Int) -> MLXArray {
        rows.reshaped(H3Constants.audioChannels, numAudioLatents, rows.dim(-1)).transposed(0, 2, 1)
    }
}

// MARK: - Packed sequence layout

/// Structural description of one packed MiniMax-H3 sequence.
/// (Not Sendable: MLXArray is not; the pipeline builds and consumes it on one task.)
public struct H3PackedSequence {
    public let sequenceLength: Int
    /// The (t, h, w) rotary coordinate of every row, float32 `(seq, 3)`
    /// (built in Double, cast once — mirroring the reference's fp64 grid + fp32 RoPE input).
    public let positionIds: MLXArray
    /// Modality tag of every row `(seq,)`, int32.
    public let tokenTags: MLXArray
    /// Positions of the video rows (conditioning rows first, then targets), int32.
    public let videoIndices: MLXArray
    /// Positions of the audio rows, int32.
    public let audioIndices: MLXArray
    /// Positions of the text rows, int32.
    public let textIndices: MLXArray
    public let numConditionVideoRows: Int
    public let numConditionAudioRows: Int
    /// Host-side copy of the per-row tags, kept for timestep-plan construction.
    public let tokenTagValues: [Int32]
    /// Host-side copies of the gather orders, conditioning rows first. A `ref2va` layout
    /// interleaves reference blocks with each other, so the timestep plan cannot assume the
    /// conditioning rows are a contiguous prefix of the sequence — only of these arrays.
    public let videoIndexValues: [Int32]
    public let audioIndexValues: [Int32]
    /// Where every row of `[text | video rows | audio rows]` lands in the packed sequence, as a
    /// gather the transformer applies to that concatenation. `nil` when the layout is already
    /// contiguous (`[text | condition video | audio | target video]`, i.e. t2va and fl2va), where
    /// plain concatenation reproduces the reference's `index_copy` scatter on its own.
    public let packedGatherOrder: MLXArray?
    /// Host-side row spans (text length etc.) used when assigning row timesteps.
    public let numTextTokens: Int
    public let numAudioRows: Int
    public let numVideoRows: Int
}

extension H3Packing {
    /// One aspect-normalized spatial rotary axis: `dim / patch` coordinates on the unit interval,
    /// scaled by 32, right endpoint excluded. Reproduces numpy `linspace(endpoint=False)` exactly:
    /// `start + i * (stop - start) / num`.
    static func spatialPositionGrid(dim: Int, patch: Int, sqrtArea: Double) -> [Double] {
        let ratio = Double(dim) / sqrtArea
        let left = (1.0 - ratio) / 2.0
        let num = dim / patch
        return (0..<num).map { (left + Double($0) * ratio / Double(num)) * H3Constants.ropeSpatialScale }
    }

    /// Rotary time of every latent frame from `origin`; non-uniform spacing 5/3 * (1,4,4,4,4) cyclic.
    static func temporalPositionGrid(numLatentFrames: Int, origin: Double) -> [Double] {
        var times = [Double](repeating: 0, count: numLatentFrames)
        var accumulated = origin
        for index in 0..<numLatentFrames {
            times[index] = accumulated
            let pattern = H3Constants.ropeFramesPerLatent[index % H3Constants.ropeFramesPerLatent.count]
            accumulated += H3Constants.ropeFrameRescale * pattern
        }
        return times
    }

    /// Total rotary time spanned by `numLatentFrames` latent frames (pairwise-summed in the
    /// reference; plain summation differs only in the last ulp and only matters for bit-exactness).
    static func temporalPositionSpan(numLatentFrames: Int) -> Double {
        (0..<numLatentFrames).reduce(0.0) {
            $0 + H3Constants.ropeFrameRescale
                * H3Constants.ropeFramesPerLatent[$1 % H3Constants.ropeFramesPerLatent.count]
        }
    }

    /// Build the `[text | keyframe conditions | target audio | target video]` layout of a
    /// t2va / fl2va request.
    ///
    /// - Parameters:
    ///   - textTokenTags: modality tag of every text row (text 1, keyframe vision blocks 0).
    ///   - keyframeAnchors: one entry per keyframe block, `"first"` or `"last"`.
    public static func buildPackedSequence(
        textTokenTags: [Int32],
        numLatentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        numAudioLatents: Int,
        patchSize: (t: Int, h: Int, w: Int),
        keyframeAnchors: [String] = []
    ) throws -> H3PackedSequence {
        let rowsPerFrame = (latentHeight / patchSize.h) * (latentWidth / patchSize.w)
        let numTextTokens = textTokenTags.count
        let numConditionRows = keyframeAnchors.count * rowsPerFrame
        let numAudioRows = numAudioLatents * H3Constants.audioChannels
        let numVideoRows = numLatentFrames * rowsPerFrame
        let sequenceLength = numTextTokens + numConditionRows + numAudioRows + numVideoRows

        let conditionStart = numTextTokens
        let audioStart = conditionStart + numConditionRows
        let videoStart = audioStart + numAudioRows

        // 1. The (t, h, w) grid. Text rows sit on the time axis at their row index; the media rows
        // continue that clock, so text length shifts the whole media timeline.
        var positions = [Double](repeating: 0, count: sequenceLength * 3)
        for index in 0..<numTextTokens { positions[index * 3] = Double(index) }

        let sqrtArea = (Double(latentHeight) * Double(latentWidth)).squareRoot()
        let heightGrid = spatialPositionGrid(dim: latentHeight, patch: patchSize.h, sqrtArea: sqrtArea)
        let widthGrid = spatialPositionGrid(dim: latentWidth, patch: patchSize.w, sqrtArea: sqrtArea)

        // frameGrid[row] = (h, w) in h-major order, one entry per patch of one latent frame.
        var frameGrid = [(Double, Double)]()
        frameGrid.reserveCapacity(rowsPerFrame)
        for h in heightGrid { for w in widthGrid { frameGrid.append((h, w)) } }

        for (blockIndex, anchor) in keyframeAnchors.enumerated() {
            let anchorTime: Double
            switch anchor {
            case "first":
                anchorTime = Double(numTextTokens)
            case "last":
                anchorTime = Double(numTextTokens)
                    + temporalPositionSpan(numLatentFrames: numLatentFrames) - H3Constants.ropeFrameRescale
            default:
                throw H3Error.invalidInput("A keyframe anchor must be 'first' or 'last', got '\(anchor)'.")
            }
            let blockStart = conditionStart + blockIndex * rowsPerFrame
            for (rowIndex, (h, w)) in frameGrid.enumerated() {
                let base = (blockStart + rowIndex) * 3
                positions[base] = anchorTime
                positions[base + 1] = h
                positions[base + 2] = w
            }
        }

        // Audio rows are channel-major and share the video's rotary clock (1 unit per latent at
        // 40 latents/s = 24 fps * 5/3). No height coordinate; width pinned to the two grid extremes.
        for channel in 0..<H3Constants.audioChannels {
            let pinnedWidth = channel == 0 ? widthGrid[0] : widthGrid[widthGrid.count - 1]
            for latent in 0..<numAudioLatents {
                let base = (audioStart + channel * numAudioLatents + latent) * 3
                positions[base] = Double(numTextTokens) + Double(latent)
                positions[base + 2] = pinnedWidth
            }
        }

        let frameTimes = temporalPositionGrid(numLatentFrames: numLatentFrames, origin: Double(numTextTokens))
        for frame in 0..<numLatentFrames {
            for (rowIndex, (h, w)) in frameGrid.enumerated() {
                let base = (videoStart + frame * rowsPerFrame + rowIndex) * 3
                positions[base] = frameTimes[frame]
                positions[base + 1] = h
                positions[base + 2] = w
            }
        }

        // 2. Row indices and modality tags.
        var videoIndexValues = [Int32]()
        videoIndexValues.reserveCapacity(numConditionRows + numVideoRows)
        videoIndexValues.append(contentsOf: (conditionStart..<audioStart).map(Int32.init))
        videoIndexValues.append(contentsOf: (videoStart..<sequenceLength).map(Int32.init))
        let audioIndexValues = (audioStart..<videoStart).map(Int32.init)
        let textIndexValues = (0..<numTextTokens).map(Int32.init)

        var tagValues = [Int32](repeating: H3Constants.videoTag, count: sequenceLength)
        for (index, tag) in textTokenTags.enumerated() { tagValues[index] = tag }
        for index in audioStart..<videoStart { tagValues[index] = H3Constants.audioTag }
        // Video rows (condition + target) already carry the video tag from the fill value.

        let positionsFloat = positions.map(Float.init)
        return H3PackedSequence(
            sequenceLength: sequenceLength,
            positionIds: MLXArray(positionsFloat, [sequenceLength, 3]),
            tokenTags: MLXArray(tagValues),
            videoIndices: MLXArray(videoIndexValues),
            audioIndices: MLXArray(audioIndexValues),
            textIndices: MLXArray(textIndexValues),
            numConditionVideoRows: numConditionRows,
            numConditionAudioRows: 0,
            tokenTagValues: tagValues,
            videoIndexValues: videoIndexValues,
            audioIndexValues: audioIndexValues,
            packedGatherOrder: nil,
            numTextTokens: numTextTokens,
            numAudioRows: numAudioRows,
            numVideoRows: numVideoRows
        )
    }

    /// Assign a timestep to every row and reduce to the transformer's
    /// `(distinct timesteps sorted ascending, per-row index)` pair.
    ///
    /// Generated video/audio rows run at their schedule's timestep, conditioning rows stay pinned,
    /// text rows inherit the video timestep (they never reach an output head).
    public static func buildRowTimesteps(
        layout: H3PackedSequence,
        videoTimestep: Float,
        audioTimestep: Float,
        conditionVideoTimestep: Float,
        conditionAudioTimestep: Float
    ) -> (timesteps: [Float], timestepIndices: MLXArray) {
        // Assigned through the gather orders rather than through sequence spans: in a `ref2va`
        // layout the reference blocks sit between the text and the generated rows in request
        // order, so "the conditioning rows" is a prefix of `videoIndices` / `audioIndices` and of
        // nothing else. Text rows keep the video timestep — they never reach an output head.
        var rowTimesteps = [Float](repeating: videoTimestep, count: layout.sequenceLength)
        for index in layout.videoIndexValues.prefix(layout.numConditionVideoRows) {
            rowTimesteps[Int(index)] = conditionVideoTimestep
        }
        for index in layout.audioIndexValues.dropFirst(layout.numConditionAudioRows) {
            rowTimesteps[Int(index)] = audioTimestep
        }
        for index in layout.audioIndexValues.prefix(layout.numConditionAudioRows) {
            rowTimesteps[Int(index)] = conditionAudioTimestep
        }

        let distinct = Array(Set(rowTimesteps)).sorted()
        var indexOf = [Float: Int32]()
        for (index, value) in distinct.enumerated() { indexOf[value] = Int32(index) }
        let rowIndices = rowTimesteps.map { indexOf[$0]! }
        return (distinct, MLXArray(rowIndices))
    }
}
