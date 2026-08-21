// ReferenceVideoProcessor.swift - Qwen3-VL video preprocessing of a ref2va video reference
// Copyright 2026 Vincent Gourbin
//
// Port of `Qwen3VLVideoProcessor._preprocess` plus MiniMax-H3's own 2 fps sampling
// (`MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames`).
//
// Two stages:
//  1. MiniMax-H3 reads a normalized 24 fps reference at 2 fps — every 12th frame, deduplicated —
//     and labels each merged frame PAIR with the mean of its two timestamps.
//  2. Qwen3-VL merges the sampled frames in groups of `temporalPatch` (2), padding the tail by
//     repeating the last one, and patchifies each group into `(gridH * gridW, 1536)` rows whose
//     per-patch layout is [channel][t][patchH][patchW] — the same 1536 an image produces, except
//     the two temporal slots hold two different frames instead of one frame twice.
//
// The vision tower attends PER TEMPORAL GROUP (`cu_seqlens` splits Qwen3-VL's vision attention on
// the temporal axis, and its rotary table carries no temporal component), so one group is one
// independent tower call — the same call an image makes. That is why this emits a list.

import Foundation
import MLX

public enum H3ReferenceVideoProcessor {
    /// Qwen3-VL's video `size` for this checkpoint (`video_preprocessor_config.json`), i.e.
    /// `smart_resize`'s pixel bounds over `t_bar * h_bar * w_bar`.
    public static let minPixels = 4096
    public static let maxPixels = 25_165_824

    /// `Qwen3VLVideoProcessor.smart_resize`, over the whole clip's `t * h * w` budget.
    /// Returns the per-frame (height, width) the processor resizes to.
    public static func smartResize(
        numFrames: Int, height: Int, width: Int, temporalPatch: Int = 2, factor: Int = 32
    ) throws -> (height: Int, width: Int) {
        guard height >= factor, width >= factor else {
            throw H3Error.invalidInput(
                "A reference video frame must be at least \(factor) px on both axes, got \(width)x\(height).")
        }
        let long = Double(max(height, width))
        let short = Double(min(height, width))
        guard long / short <= 200 else {
            throw H3Error.invalidInput("A reference video's aspect ratio must stay under 200:1.")
        }
        func round(_ value: Double) -> Int {
            Int((value / Double(factor)).rounded(.toNearestOrEven)) * factor
        }
        var barHeight = round(Double(height))
        var barWidth = round(Double(width))
        let barFrames = (numFrames + temporalPatch - 1) / temporalPatch * temporalPatch
        let budget = Double(barFrames) * Double(barHeight) * Double(barWidth)
        let pixels = Double(numFrames) * Double(height) * Double(width)
        if budget > Double(maxPixels) {
            let beta = (pixels / Double(maxPixels)).squareRoot()
            barHeight = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            barWidth = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if budget < Double(minPixels) {
            let beta = (Double(minPixels) / pixels).squareRoot()
            barHeight = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            barWidth = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (barHeight, barWidth)
    }

    /// The frames the conditioner sees, and the timestamp label of every merged vision block.
    ///
    /// `stride = fps / sampleFPS`; indices are `round(cursor)` walked by `stride`, deduplicated.
    /// A block's timestamp is the MEAN of its group's timestamps, which the reference renders with
    /// `"{:.1f}"` — Python's round-half-to-EVEN — so the first block of a 2 fps pair reads
    /// `<0.2 seconds>`, not `<0.3 seconds>`.
    public static func sampleConditionFrames(
        frameCount: Int, fps: Double = Double(H3Constants.fps),
        sampleFPS: Double = H3ReferenceNormalizer.conditionerSampleFPS, temporalPatch: Int = 2
    ) throws -> (indices: [Int], blockTimestamps: [Double]) {
        let step = fps / sampleFPS
        var indices = [Int]()
        var cursor = 0.0
        // Python `round`, i.e. half-to-even. At the released 24 -> 2 fps the stride is exactly 12
        // and no tie can occur, but a fine-tuned `sampleFPS` would land on them.
        while Int(cursor.rounded(.toNearestOrEven)) < frameCount {
            let index = Int(cursor.rounded(.toNearestOrEven))
            if indices.isEmpty || index > indices[indices.count - 1] { indices.append(index) }
            cursor += step
        }
        guard indices.count >= temporalPatch else {
            let minimum = Int((Double(temporalPatch - 1) * step).rounded(.toNearestOrEven)) + 1
            throw H3Error.invalidInput(
                "A reference video is read at \(sampleFPS) fps and its sampled frames are merged in "
                    + "groups of \(temporalPatch), so it must run at least \(minimum) frames at "
                    + "\(fps) fps, got \(frameCount).")
        }

        var timestamps = (0..<indices.count).map { Double($0) / sampleFPS }
        let tail = (temporalPatch - timestamps.count % temporalPatch) % temporalPatch
        timestamps.append(contentsOf: [Double](repeating: timestamps[timestamps.count - 1], count: tail))
        var blockTimestamps = [Double]()
        for start in stride(from: 0, to: timestamps.count, by: temporalPatch) {
            blockTimestamps.append((timestamps[start] + timestamps[start + temporalPatch - 1]) / 2)
        }
        return (indices, blockTimestamps)
    }

    /// `"<{t:.1f} seconds>"` with Python's round-half-to-even, which is what labels a vision block.
    public static func timestampLabel(_ seconds: Double) -> String {
        let scaled = (seconds * 10).rounded(.toNearestOrEven) / 10
        return String(format: "<%.1f seconds>", scaled)
    }

    /// One prepared reference video: its per-block patch rows and the grid they carry.
    public struct Prepared {
        /// One `(gridH * gridW, 1536)` tensor per merged frame pair, in order.
        public let blocks: [MLXArray]
        public let gridH: Int
        public let gridW: Int
        /// `"<t seconds>"` label of every block, positional with `blocks`.
        public let blockTimestamps: [Double]
    }

    /// Sample, resize and patchify a normalized 24 fps reference video for the conditioner.
    public static func prepare(
        frames: [H3KeyframeImage], patchSize: Int = 16, mergeSize: Int = 2, temporalPatch: Int = 2
    ) throws -> Prepared {
        guard let first = frames.first else {
            throw H3Error.invalidInput("A reference video must carry at least one frame.")
        }
        let (indices, blockTimestamps) = try sampleConditionFrames(
            frameCount: frames.count, temporalPatch: temporalPatch)

        let (height, width) = try smartResize(
            numFrames: indices.count, height: first.height, width: first.width,
            temporalPatch: temporalPatch, factor: patchSize * mergeSize)
        // Frames normalized onto their own canvas are already a multiple of 32 and, below roughly
        // 12.5 s of reference, already inside `maxPixels` — so this is a pass-through on the
        // common path and only a long reference at the largest canvas is actually downscaled.
        var sampled = indices.map { index -> H3KeyframeImage in
            let frame = frames[index]
            return frame.width == width && frame.height == height
                ? frame
                : frame.resized(toWidth: width, toHeight: height, filter: .bicubic)
        }
        // Tail padded to the temporal patch by repeating the last sampled frame.
        let tail = (temporalPatch - sampled.count % temporalPatch) % temporalPatch
        if tail > 0 {
            sampled.append(contentsOf: [H3KeyframeImage](repeating: sampled[sampled.count - 1], count: tail))
        }
        guard sampled.count / temporalPatch == blockTimestamps.count else {
            throw H3Error.invalidInput(
                "The processor merged the reference into \(sampled.count / temporalPatch) vision blocks "
                    + "but MiniMax-H3 labels \(blockTimestamps.count) of them.")
        }

        let (gridH, gridW) = (height / patchSize, width / patchSize)
        var blocks = [MLXArray]()
        blocks.reserveCapacity(sampled.count / temporalPatch)
        for group in stride(from: 0, to: sampled.count, by: temporalPatch) {
            blocks.append(
                patchifyGroup(
                    Array(sampled[group..<(group + temporalPatch)]),
                    patchSize: patchSize, mergeSize: mergeSize))
        }
        return Prepared(blocks: blocks, gridH: gridH, gridW: gridW, blockTimestamps: blockTimestamps)
    }

    /// One temporal group -> `(gridH * gridW, 3 * temporalPatch * patch * patch)` rows in the
    /// processor's block-major token order, per-patch layout [channel][t][patchH][patchW].
    static func patchifyGroup(
        _ group: [H3KeyframeImage], patchSize: Int, mergeSize: Int
    ) -> MLXArray {
        let (height, width) = (group[0].height, group[0].width)
        let (gridH, gridW) = (height / patchSize, width / patchSize)
        let (blocksH, blocksW) = (gridH / mergeSize, gridW / mergeSize)
        let temporalPatch = group.count

        // (T, H, W, C) -> (T, C, H, W) normalized
        var x = MLXArray(group.flatMap(\.pixels), [temporalPatch, height, width, 3]).asType(.float32)
        x = (x / 255.0 - 0.5) / 0.5
        x = x.transposed(0, 3, 1, 2)
        x = x.reshaped(temporalPatch, 3, blocksH, mergeSize, patchSize, blocksW, mergeSize, patchSize)
        // -> [hBlock, wBlock, mergeH, mergeW, C, t, patchH, patchW]
        x = x.transposed(2, 5, 3, 6, 1, 0, 4, 7)
        return x.reshaped(gridH * gridW, 3 * temporalPatch * patchSize * patchSize)
    }
}
