// H3MediaDecoder.swift - reference media ingestion (AVFoundation, replacing the reference's PyAV)
// Copyright 2026 Vincent Gourbin
//
// Port of `references.py`'s `_decode_video_file` / `_decode_audio_file`. The contract is the rates:
// a video comes back with the frame rate its container reports and a waveform with its own sample
// rate, because MiniMax-H3 resamples both onto its own clocks and a lost rate conditions the
// request at the wrong speed with nothing to raise about it.
//
// Frames come back as `uint8` RGB at the container's resolution, with the display-matrix rotation
// undone (snapped to the nearest quarter turn) and a non-square pixel aspect ratio LEFT ALONE — the
// reference resolves a reference's canvas from its display geometry and explicitly refuses to
// correct a sample aspect ratio, so correcting it here would be untested guesswork.
//
// Pixel-level parity note: RGB comes out of VideoToolbox's colour conversion, not swscale's, so a
// decoded frame can differ from PyAV's by a few LSBs. That is the same out-of-contract boundary the
// keyframe path already documents; parity probes feed decoded pixels rather than files.

import AVFoundation
import CoreVideo
import Foundation

public enum H3MediaDecoder {
    // MARK: - Display-matrix rotation

    /// The clockwise quarter turns that put a decoded frame upright, from an
    /// `AVAssetTrack.preferredTransform`.
    ///
    /// FFmpeg's `av_display_rotation_get` — which is what the reference reads as `frame.rotation` —
    /// is `-atan2(b, a)` in degrees over the ISO display matrix, whose `a`/`b` are a
    /// `CGAffineTransform`'s `a`/`b`. The reference then snaps to `turns = round(rotation / 90) % 4`
    /// and rotates by `np.rot90(k: -turns)`, i.e. clockwise by `turns`.
    public static func quarterTurns(fromPreferredTransform transform: CGAffineTransform) -> Int {
        let rotation = -atan2(Double(transform.b), Double(transform.a)) * 180.0 / Double.pi
        guard rotation.isFinite else { return 0 }
        let turns = Int((rotation / 90.0).rounded(.toNearestOrAwayFromZero))
        return ((turns % 4) + 4) % 4
    }

    // MARK: - Video

    /// Decode a video file into reference frames plus its soundtrack, both at their own rates.
    ///
    /// - Parameters:
    ///   - maxDuration: stop after this many seconds of source material. The 24 fps resample maps
    ///     output slot `j` to a source frame index that depends only on `j`, so cutting the source
    ///     beyond what the generated frame count can consume cannot change the normalized result —
    ///     which is what keeps a long 4K clip from being held in memory whole.
    ///   - fpsOverride: use this rate instead of the container's, for a container whose metadata
    ///     is wrong.
    public static func decodeVideo(
        at url: URL, maxDuration: Double? = nil, fpsOverride: Double? = nil
    ) async throws -> H3VideoReference {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw H3Error.invalidInput("No video stream to decode in \(url.lastPathComponent).")
        }
        let (nominalRate, transform) = try await track.load(.nominalFrameRate, .preferredTransform)
        let fps = fpsOverride ?? Double(nominalRate)
        guard fps > 0 else {
            throw H3Error.invalidInput(
                "\(url.lastPathComponent) reports no frame rate; pass an explicit reference fps.")
        }
        let turns = quarterTurns(fromPreferredTransform: transform)

        let reader = try AVAssetReader(asset: asset)
        if let maxDuration {
            // One extra second of slack: the last output slot may read a frame just past the cut.
            reader.timeRange = CMTimeRange(
                start: .zero, duration: CMTime(seconds: maxDuration + 1.0, preferredTimescale: 600))
        }
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw H3Error.invalidInput("Cannot read the video stream of \(url.lastPathComponent).")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw H3Error.invalidInput(
                "Cannot start reading \(url.lastPathComponent): \(reader.error?.localizedDescription ?? "unknown").")
        }

        var frames = [H3KeyframeImage]()
        while let sample = output.copyNextSampleBuffer() {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            frames.append(try rgbFrame(from: buffer).rotatedClockwise(quarterTurns: turns))
        }
        if reader.status == .failed {
            throw H3Error.invalidInput(
                "Decoding \(url.lastPathComponent) failed: \(reader.error?.localizedDescription ?? "unknown").")
        }
        guard !frames.isEmpty else {
            throw H3Error.invalidInput("No video frames to decode in \(url.lastPathComponent).")
        }

        // The soundtrack is read in a second pass, as the reference does after draining the frames.
        let soundtrack = try await decodeAudioIfPresent(asset: asset, maxDuration: maxDuration)
        return H3VideoReference(frames: frames, fps: fps, audio: soundtrack)
    }

    /// One BGRA pixel buffer as interleaved RGB. Throws rather than yielding a black frame if
    /// the buffer cannot be mapped: the frame count would still line up, so a silently black
    /// frame would be conditioned on with nothing downstream able to notice.
    private static func rgbFrame(from buffer: CVImageBuffer) throws -> H3KeyframeImage {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw H3Error.invalidInput("A decoded video frame could not be mapped into memory.")
        }
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        do {
            let source = base.assumingMemoryBound(to: UInt8.self)
            pixels.withUnsafeMutableBufferPointer { destination in
                for y in 0..<height {
                    let row = source + y * stride
                    var write = y * width * 3
                    for x in 0..<width {
                        let pixel = row + x * 4  // BGRA
                        destination[write] = pixel[2]
                        destination[write + 1] = pixel[1]
                        destination[write + 2] = pixel[0]
                        write += 3
                    }
                }
            }
        }
        return H3KeyframeImage(width: width, height: height, pixels: pixels)
    }

    // MARK: - Audio

    /// Whether a container carries an audio stream, without decoding any of it. The prompt
    /// rewriter needs this: a video reference that carries a soundtrack takes an `"<Audio j>"`
    /// label as well as its `"<Video k>"`, and the labels have to match what the presentation
    /// will emit.
    public static func hasAudioTrack(at url: URL) async throws -> Bool {
        try await !AVURLAsset(url: url).loadTracks(withMediaType: .audio).isEmpty
    }

    /// Decode an audio file (or a video file's soundtrack) at the sample rate and channel layout
    /// its container reports, float32.
    public static func decodeAudio(at url: URL, maxDuration: Double? = nil) async throws -> H3AudioBuffer {
        let asset = AVURLAsset(url: url)
        guard let buffer = try await decodeAudioIfPresent(asset: asset, maxDuration: maxDuration) else {
            throw H3Error.invalidInput("No audio stream to decode in \(url.lastPathComponent).")
        }
        return buffer
    }

    static func decodeAudioIfPresent(asset: AVURLAsset, maxDuration: Double?) async throws -> H3AudioBuffer? {
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else { return nil }
        let descriptions = try await track.load(.formatDescriptions)
        guard let format = descriptions.first,
              let basic = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee
        else {
            throw H3Error.invalidInput("Cannot read the audio format of the reference.")
        }
        let sampleRate = Int(basic.mSampleRate.rounded())
        let channels = Int(basic.mChannelsPerFrame)
        guard sampleRate > 0, channels > 0 else {
            throw H3Error.invalidInput("The reference soundtrack reports no sample rate or channels.")
        }

        let reader = try AVAssetReader(asset: asset)
        if let maxDuration {
            reader.timeRange = CMTimeRange(
                start: .zero, duration: CMTime(seconds: maxDuration + 1.0, preferredTimescale: 600))
        }
        // Native rate and channel layout: the reference's PyAV resampler converts the *format*
        // only, and a mono soundtrack is upmixed later, by the normalization — not here.
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw H3Error.invalidInput("Cannot read the audio stream of the reference.")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw H3Error.invalidInput(
                "Cannot start reading the reference soundtrack: \(reader.error?.localizedDescription ?? "unknown").")
        }

        var interleaved = [Float]()
        while let sample = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / MemoryLayout<Float>.size)
            chunk.withUnsafeMutableBytes { destination in
                _ = CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: length, destination: destination.baseAddress!)
            }
            interleaved.append(contentsOf: chunk)
        }
        if reader.status == .failed {
            throw H3Error.invalidInput(
                "Decoding the reference soundtrack failed: \(reader.error?.localizedDescription ?? "unknown").")
        }
        guard !interleaved.isEmpty else { return nil }

        let frameCount = interleaved.count / channels
        var planar = [Float](repeating: 0, count: frameCount * channels)
        for channel in 0..<channels {
            for frame in 0..<frameCount {
                planar[channel * frameCount + frame] = interleaved[frame * channels + channel]
            }
        }
        // Beyond stereo, MiniMax-H3 has no convention: keep the first two channels, which is what a
        // stereo reference would have carried.
        if channels > 2 {
            return H3AudioBuffer(
                channels: 2, frameCount: frameCount,
                samples: Array(planar[0..<(2 * frameCount)]), sampleRate: sampleRate)
        }
        return H3AudioBuffer(
            channels: channels, frameCount: frameCount, samples: planar, sampleRate: sampleRate)
    }
}
