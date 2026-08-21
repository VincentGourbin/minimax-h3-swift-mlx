// H3References.swift - the ordered reference list of a MiniMax-H3 `ref2va` request
// Copyright 2026 Vincent Gourbin
//
// Port of diffusers `modular_pipelines/minimax_h3/references.py` (the three dataclasses) and of
// `MiniMaxH3Ref2VASetupStep` (validation + normalization) from `before_encoder.py`.
//
// Two invariants carry the whole mode:
//
//  1. The list is ORDERED and the order is the request. It fixes the `<Picture i>` / `<Audio j>` /
//     `<Video k>` labels of the presentation AND it advances the shared audio/video rotary clock,
//     so a reordered list is a different request. Nothing here sorts or groups by modality.
//  2. RATES TRAVEL WITH MEDIA. A video carries its `fps`, a waveform its `sampleRate`. H3 resamples
//     everything onto its own 24 fps / 32 kHz; a rate lost on the way in conditions the request at
//     the wrong speed, silently, with nothing to raise about it. `nil` means "already at H3's own
//     rate" and leaves the media untouched.

import Foundation

// MARK: - Media buffers

/// A decoded waveform: `(channels, frameCount)` float32, channel-major (channel `c` occupies
/// `samples[c * frameCount ..< (c + 1) * frameCount]`), which is torch's `(channels, num_samples)`.
public struct H3AudioBuffer: Sendable, Equatable {
    public var channels: Int
    /// Samples per channel.
    public var frameCount: Int
    /// `channels * frameCount` values, channel-major.
    public var samples: [Float]
    /// The rate `samples` carries. `nil` = already the audio VAE's own (32 kHz), left untouched.
    public var sampleRate: Int?

    public init(channels: Int, frameCount: Int, samples: [Float], sampleRate: Int?) {
        precondition(samples.count == channels * frameCount)
        self.channels = channels
        self.frameCount = frameCount
        self.samples = samples
        self.sampleRate = sampleRate
    }

    /// One channel's samples.
    public func channel(_ index: Int) -> ArraySlice<Float> {
        samples[(index * frameCount)..<((index + 1) * frameCount)]
    }

    public var duration: Double {
        Double(frameCount) / Double(sampleRate ?? H3Constants.audioSampleRate)
    }
}

// MARK: - The three reference types

/// A subject, style or scene reference: at most 9 per request. It never binds the generated
/// geometry — it is encoded at a short edge of its own (2048), whatever canvas the request runs at.
public struct H3ImageReference: Sendable {
    public var image: H3KeyframeImage
    public init(image: H3KeyframeImage) { self.image = image }
}

/// A motion and camera reference: at most 3 per request, conditioned on together with its own
/// soundtrack. `fps` is what places its vision blocks on the conditioner's 2 fps grid and what the
/// 24 fps resample is computed from — the field to get right when the frames came from a file.
public struct H3VideoReference: Sendable {
    public var frames: [H3KeyframeImage]
    public var fps: Double
    /// This video's soundtrack, conditioned on as the reference's own rather than as a reference of
    /// its own. Left out, the reference conditions on motion alone.
    public var audio: H3AudioBuffer?

    public init(frames: [H3KeyframeImage], fps: Double = Double(H3Constants.fps), audio: H3AudioBuffer? = nil) {
        self.frames = frames
        self.fps = fps
        self.audio = audio
    }
}

/// A voice or music reference: at most 3 per request, and never on its own — it has to be paired
/// with at least one image or video reference. It never reaches the conditioner and is encoded by
/// the audio VAE alone.
public struct H3AudioReference: Sendable {
    public var audio: H3AudioBuffer
    public init(audio: H3AudioBuffer) { self.audio = audio }
}

public enum H3ReferenceKind: String, Sendable {
    case image, video, audio
}

public enum H3Reference: Sendable {
    case image(H3ImageReference)
    case video(H3VideoReference)
    case audio(H3AudioReference)

    public var kind: H3ReferenceKind {
        switch self {
        case .image: return .image
        case .video: return .video
        case .audio: return .audio
        }
    }

    /// Whether this reference contributes audio rows, i.e. whether it carries a waveform.
    /// A video reference without a soundtrack does not; an audio reference always does.
    public var hasAudio: Bool {
        switch self {
        case .image: return false
        case .video(let reference): return reference.audio != nil
        case .audio: return true
        }
    }

    /// The waveform this reference contributes, if any.
    public var audioBuffer: H3AudioBuffer? {
        switch self {
        case .image: return nil
        case .video(let reference): return reference.audio
        case .audio(let reference): return reference.audio
        }
    }
}

// MARK: - Validation + normalization (MiniMaxH3Ref2VASetupStep)

public enum H3ReferenceNormalizer {
    /// The limits MiniMax-H3 documents for the released checkpoint. They bound validation only.
    public static let maxImages = 9
    public static let maxVideos = 3
    public static let maxAudios = 3
    public static let maxReferences = 12

    /// Image references are encoded at high detail — a short edge of their own, upscaling included
    /// and with *no* area cap — unlike video references and the target, which share the canvas rule.
    public static let referenceImageShortEdge = 2048

    /// The rate the conditioner reads a reference video at.
    public static let conditionerSampleFPS = 2.0

    // MARK: Validation

    /// Enforce the reference-list rules with errors, never by clamping.
    public static func validate(_ references: [H3Reference]) throws {
        guard !references.isEmpty else {
            throw H3Error.invalidInput(
                "`ref2va` needs at least one reference; use the t2va workflow for text-only requests.")
        }
        let kinds = references.map(\.kind)
        for (kind, limit) in [
            (H3ReferenceKind.image, maxImages), (.video, maxVideos), (.audio, maxAudios),
        ] {
            let count = kinds.filter { $0 == kind }.count
            guard count <= limit else {
                throw H3Error.invalidInput(
                    "MiniMax-H3 accepts at most \(limit) \(kind.rawValue) references, got \(count).")
            }
        }
        guard kinds.count <= maxReferences else {
            throw H3Error.invalidInput(
                "MiniMax-H3 accepts at most \(maxReferences) references in total, got \(kinds.count).")
        }
        guard kinds.contains(where: { $0 != .audio }) else {
            throw H3Error.invalidInput(
                "An audio reference has to be paired with at least one image or video reference "
                    + "and cannot be used on its own.")
        }
    }

    // MARK: Frame-rate resampling

    /// The source frame index behind every 24 fps output slot, capped at `limit` slots.
    ///
    /// `ffmpeg`'s `fps` filter, which is what the reference implementation decoded through: every
    /// frame is held until the slot of the next one, `slots[i] = floor(i * scale + 0.5)`, and the
    /// last one until the slot the stream's end rounds to. Whole frames only, never blended.
    static func resampleFrameIndices(
        sourceCount: Int, fps: Double, targetFPS: Double, limit: Int
    ) -> [Int] {
        precondition(sourceCount > 0 && limit >= 0)
        if fps == targetFPS { return Array(0..<min(sourceCount, limit)) }
        let scale = targetFPS / fps
        func slot(_ index: Int) -> Int { Int((Double(index) * scale + 0.5).rounded(.down)) }
        let end = slot(sourceCount)
        var indices = [Int]()
        indices.reserveCapacity(min(end, limit))
        for source in 0..<sourceCount {
            let start = slot(source)
            let stop = source + 1 < sourceCount ? slot(source + 1) : end
            for _ in start..<stop {
                if indices.count == limit { return indices }
                indices.append(source)
            }
        }
        return indices
    }

    /// How many source frames at `fps` a 24 fps output of `numFrames` slots can possibly need.
    /// Decoding beyond this cannot change the truncated output, which is what lets a long or
    /// high-resolution clip be decoded without holding all of it in memory.
    public static func sourceFramesNeeded(numFrames: Int, fps: Double, targetFPS: Double = Double(H3Constants.fps)) -> Int {
        guard fps > 0, targetFPS > 0, fps != targetFPS else { return numFrames }
        return Int((Double(numFrames) * fps / targetFPS).rounded(.up)) + 2
    }

    // MARK: Per-modality normalization

    /// Normalize a video reference's frames: onto 24 fps by dropping and duplicating whole frames,
    /// truncated to the generated frame count, then onto the canvas its OWN aspect ratio resolves
    /// to — the same rule the target follows, unlike an image reference.
    ///
    /// The two passes are in the reference implementation's `ffmpeg` decode order: the constant
    /// frame-rate resample first, the LANCZOS rescale second. Frames already at 24 fps and already
    /// on their canvas flow through untouched, which is the parity-exact route (the reference
    /// rescaled with `ffmpeg`'s own LANCZOS while decoding, so only frames decoded at the canvas
    /// reproduce its pixels bit for bit).
    public static func normalizeVideoFrames(
        _ frames: [H3KeyframeImage], fps: Double, numFrames: Int
    ) throws -> [H3KeyframeImage] {
        guard !frames.isEmpty else {
            throw H3Error.invalidInput("A reference video must carry at least one frame.")
        }
        guard fps > 0 else {
            throw H3Error.invalidInput("A reference video must have a positive frame rate, got \(fps).")
        }
        let indices = resampleFrameIndices(
            sourceCount: frames.count, fps: fps, targetFPS: Double(H3Constants.fps), limit: numFrames)
        let first = frames[0]
        let (height, width) = try H3Geometry.resolveCanvasSize(
            aspectWidth: Double(first.width), aspectHeight: Double(first.height))

        // One resample per distinct source frame: the 24 fps grid duplicates frames, and rescaling
        // the same pixels twice is pure waste.
        var rescaled = [Int: H3KeyframeImage]()
        return indices.map { source in
            if let cached = rescaled[source] { return cached }
            let frame = frames[source]
            let result = frame.width == width && frame.height == height
                ? frame
                : frame.resizedLanczos(toWidth: width, toHeight: height)
            rescaled[source] = result
            return result
        }
    }

    /// Normalize a reference soundtrack onto the audio VAE's sample rate, as stereo.
    ///
    /// Truncate at the NATIVE rate, upmix mono by repeating the channel, then resample ONCE — the
    /// reference implementation's order, and not interchangeable: truncating after resampling
    /// rounds a different number of samples away.
    public static func normalizeAudio(
        _ buffer: H3AudioBuffer, maxDuration: Double
    ) throws -> H3AudioBuffer {
        guard buffer.channels == 1 || buffer.channels == 2 else {
            throw H3Error.invalidInput(
                "A reference soundtrack must be a mono or stereo `(channels, num_samples)` waveform, "
                    + "got \(buffer.channels) channels.")
        }
        let sampleRate = buffer.sampleRate ?? H3Constants.audioSampleRate
        guard sampleRate > 0 else {
            throw H3Error.invalidInput("A reference soundtrack must have a positive sample rate.")
        }

        let keep = min(buffer.frameCount, max(0, Int(maxDuration * Double(sampleRate))))
        guard keep > 0 else {
            throw H3Error.invalidInput(
                "A reference soundtrack truncated to \(maxDuration) s at \(sampleRate) Hz is empty.")
        }
        var channels = (0..<buffer.channels).map { Array(buffer.channel($0).prefix(keep)) }
        if channels.count == 1 { channels.append(channels[0]) }  // mono upmixed by channel repeat

        if sampleRate == H3Constants.audioSampleRate {
            return H3AudioBuffer(
                channels: 2, frameCount: keep, samples: channels[0] + channels[1],
                sampleRate: H3Constants.audioSampleRate)
        }
        let resampled = try channels.map {
            try H3Resampler.resample($0, from: sampleRate, to: H3Constants.audioSampleRate)
        }
        return H3AudioBuffer(
            channels: 2, frameCount: resampled[0].count, samples: resampled[0] + resampled[1],
            sampleRate: H3Constants.audioSampleRate)
    }

    /// The resolution a reference image is encoded at: its own 2048 pixel short edge — upscaling
    /// included, with **no** area cap, deliberately different from the canvas rule the target and
    /// video references share. Split from the resample so the geometry can be checked without
    /// paying for a 2048-edge LANCZOS pass.
    public static func referenceImageSize(width: Int, height: Int) throws -> (height: Int, width: Int) {
        guard width > 0, height > 0 else {
            throw H3Error.invalidInput("A reference image must have a positive size.")
        }
        guard width <= 4 * height, height <= 4 * width else {
            throw H3Error.invalidInput(
                "A reference image must be within 1:4 and 4:1, got \(width)x\(height).")
        }
        let multiple = H3Constants.canvasMultiple
        let scale = Double(referenceImageShortEdge) / Double(min(width, height))
        // Python `round` is half-to-even, and a 2048 short edge lands on exact halves often enough.
        func snap(_ value: Int) -> Int {
            max(multiple, Int((Double(value) * scale / Double(multiple)).rounded(.toNearestOrEven)) * multiple)
        }
        return (snap(height), snap(width))
    }

    /// Normalize a reference image onto `referenceImageSize`.
    public static func normalizeImage(_ image: H3KeyframeImage) throws -> H3KeyframeImage {
        let (height, width) = try referenceImageSize(width: image.width, height: image.height)
        if image.width == width && image.height == height { return image }
        return image.resizedLanczos(toWidth: width, toHeight: height)
    }

    // MARK: The setup step

    /// Validate a `ref2va` request's references and normalize them onto MiniMax-H3's own rates and
    /// resolutions, in packed order. `numFrames` must already be the aligned `17n + 5` count: a
    /// soundtrack is truncated to the generated duration, so the frame count has to be final first.
    public static func normalize(_ references: [H3Reference], numFrames: Int) throws -> [H3Reference] {
        try validate(references)
        guard numFrames % H3Constants.framesPerChunk == H3Constants.latentsPerChunk else {
            throw H3Error.invalidInput(
                "`numFrames` must be the aligned `17 * n + 5` count before references are normalized, "
                    + "got \(numFrames).")
        }
        let maxDuration = Double(numFrames) / Double(H3Constants.fps)

        return try references.map { reference in
            switch reference {
            case .image(let entry):
                return .image(H3ImageReference(image: try normalizeImage(entry.image)))
            case .video(let entry):
                let waveform = try entry.audio.map { try normalizeAudio($0, maxDuration: maxDuration) }
                let frames = try normalizeVideoFrames(
                    entry.frames, fps: entry.fps, numFrames: numFrames)
                // The conditioner reads a reference at 2 fps and merges the sampled frames in
                // pairs, so a clip too short to yield two of them cannot be presented. The
                // reference raises this too, but only once the conditioner runs — surfacing it
                // here costs nothing and saves a tokenizer load.
                _ = try H3ReferenceVideoProcessor.sampleConditionFrames(frameCount: frames.count)
                return .video(
                    H3VideoReference(
                        frames: frames,
                        fps: Double(H3Constants.fps),
                        audio: waveform))
            case .audio(let entry):
                return .audio(
                    H3AudioReference(audio: try normalizeAudio(entry.audio, maxDuration: maxDuration)))
            }
        }
    }
}
