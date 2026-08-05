// VideoExporter.swift - MP4 muxing (H.264 video + AAC stereo) via AVFoundation
// Copyright 2026 Vincent Gourbin
//
// Non-realtime AVAssetWriterInputs only pump data through requestMediaDataWhenReady — polling
// isReadyForMoreMediaData without it deadlocks once the input's internal buffer fills (see
// docs/knowledge/pitfalls/avassetwriter-requestmediadata.md). All appends happen inside the
// ready callbacks.

import AVFoundation
import CoreVideo
import Foundation
import MLX

public enum H3VideoExporter {
    private final class ExportProgress: @unchecked Sendable {
        var frameIndex = 0
        var finished = false
    }

    /// Thread-safe completion latch: `finish` may run before or after `wait` is called.
    private final class TrackCompletion: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<Void, Error>?
        private var continuation: CheckedContinuation<Void, Error>?

        func finish(_ error: Error?) {
            lock.lock()
            let waiter = continuation
            continuation = nil
            if result == nil { result = error.map { .failure($0) } ?? .success(()) }
            let outcome = result!
            lock.unlock()
            waiter?.resume(with: outcome)
        }

        func wait() async throws {
            try await withCheckedThrowingContinuation { (candidate: CheckedContinuation<Void, Error>) in
                lock.lock()
                if let outcome = result {
                    lock.unlock()
                    candidate.resume(with: outcome)
                } else {
                    continuation = candidate
                    lock.unlock()
                }
            }
        }
    }

    /// Write frames (F, H, W, 3) uint8 RGB at `fps` plus a stereo float32 waveform (2, samples)
    /// at `sampleRate` into an MP4.
    public static func export(
        frames: MLXArray,
        audio: MLXArray,
        fps: Int,
        sampleRate: Int,
        to outputURL: URL
    ) async throws {
        let (frameCount, height, width) = (frames.dim(0), frames.dim(1), frames.dim(2))
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: outputURL)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

        let videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 12_000_000],
            ]
        )
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        writer.add(videoInput)

        let audioInput = AVAssetWriterInput(
            mediaType: .audio,
            outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 192_000,
            ]
        )
        audioInput.expectsMediaDataInRealTime = false
        writer.add(audioInput)

        H3Debug.log("export: begin \(frameCount) frames \(width)x\(height)")
        guard writer.startWriting() else {
            throw H3Error.exportFailed(writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)
        H3Debug.log("export: writer status \(writer.status.rawValue), videoReady \(videoInput.isReadyForMoreMediaData)")

        // Both tracks MUST be fed concurrently: an MP4 writer interleaves samples across its
        // inputs and stops accepting video that runs more than its interleave window (~2-3 s)
        // ahead of audio. Feeding all video before any audio deadlocks past ~60 frames.
        let pixels = frames.asArray(UInt8.self)
        let frameBytes = height * width * 3
        let progress = ExportProgress()
        let videoQueue = DispatchQueue(label: "h3.export.video")
        let audioQueue = DispatchQueue(label: "h3.export.audio")
        let sampleBuffer = try makeAudioSampleBuffer(audio: audio, sampleRate: sampleRate)

        // 1. Register the audio pump first (single sample; completion tracked out-of-band).
        let audioCompletion = TrackCompletion()
        let audioProgress = ExportProgress()
        audioInput.requestMediaDataWhenReady(on: audioQueue) {
            guard !audioProgress.finished, audioInput.isReadyForMoreMediaData else { return }
            audioProgress.finished = true
            let appended = audioInput.append(sampleBuffer)
            audioInput.markAsFinished()
            if appended {
                audioCompletion.finish(nil)
            } else {
                audioCompletion.finish(
                    H3Error.exportFailed(
                        "Audio append failed: \(writer.error?.localizedDescription ?? "?")"))
            }
        }

        // 2. Stream the video frames.
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            videoInput.requestMediaDataWhenReady(on: videoQueue) {
                guard !progress.finished else { return }
                while videoInput.isReadyForMoreMediaData {
                    if progress.frameIndex == frameCount {
                        progress.finished = true
                        videoInput.markAsFinished()
                        continuation.resume()
                        return
                    }
                    do {
                        let buffer = try makePixelBuffer(
                            adaptor: adaptor, pixels: pixels,
                            offset: progress.frameIndex * frameBytes,
                            width: width, height: height
                        )
                        let time = CMTime(
                            value: CMTimeValue(progress.frameIndex), timescale: CMTimeScale(fps))
                        guard adaptor.append(buffer, withPresentationTime: time) else {
                            throw H3Error.exportFailed(
                                "Frame append failed: \(writer.error?.localizedDescription ?? "?")")
                        }
                        progress.frameIndex += 1
                    } catch {
                        progress.finished = true
                        videoInput.markAsFinished()
                        continuation.resume(throwing: error)
                        return
                    }
                }
            }
        }
        H3Debug.log("export: video track done")

        // 3. Wait for the audio pump (usually already finished long before the video).
        try await audioCompletion.wait()
        H3Debug.log("export: audio track done")
        await writer.finishWriting()
        if writer.status != .completed {
            throw H3Error.exportFailed(writer.error?.localizedDescription ?? "Writer did not complete")
        }
    }

    // MARK: - Buffers

    private static func makePixelBuffer(
        adaptor: AVAssetWriterInputPixelBufferAdaptor,
        pixels: [UInt8],
        offset: Int,
        width: Int,
        height: Int
    ) throws -> CVPixelBuffer {
        guard let pool = adaptor.pixelBufferPool else {
            throw H3Error.exportFailed("No pixel buffer pool (writer failed early?)")
        }
        var maybeBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer)
        guard let buffer = maybeBuffer else { throw H3Error.exportFailed("Pixel buffer allocation failed") }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let base = CVPixelBufferGetBaseAddress(buffer)!
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        // Raw-pointer walk: a captured-array subscript in this hot loop costs ~100x (bounds
        // checks + closure-box traffic), turning export into seconds per frame.
        pixels.withUnsafeBufferPointer { pixelBuffer in
            let source = pixelBuffer.baseAddress! + offset
            for row in 0..<height {
                let rowPointer = base.advanced(by: row * stride).assumingMemoryBound(to: UInt8.self)
                var read = source + row * width * 3
                var write = 0
                for _ in 0..<width {
                    rowPointer[write] = read[2]
                    rowPointer[write + 1] = read[1]
                    rowPointer[write + 2] = read[0]
                    rowPointer[write + 3] = 255
                    read += 3
                    write += 4
                }
            }
        }
        return buffer
    }

    private static func makeAudioSampleBuffer(audio: MLXArray, sampleRate: Int) throws -> CMSampleBuffer {
        let left = audio[0].asArray(Float.self)
        let right = audio[1].asArray(Float.self)
        let sampleCount = min(left.count, right.count)
        var interleaved = [Float](repeating: 0, count: sampleCount * 2)
        for index in 0..<sampleCount {
            interleaved[index * 2] = left[index]
            interleaved[index * 2 + 1] = right[index]
        }

        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 8,
            mFramesPerPacket: 1,
            mBytesPerFrame: 8,
            mChannelsPerFrame: 2,
            mBitsPerChannel: 32,
            mReserved: 0
        )
        var maybeFormat: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(
            allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil,
            magicCookieSize: 0, magicCookie: nil, extensions: nil,
            formatDescriptionOut: &maybeFormat
        )
        guard let format = maybeFormat else { throw H3Error.exportFailed("Audio format creation failed") }

        let byteCount = interleaved.count * MemoryLayout<Float>.size
        var maybeBlock: CMBlockBuffer?
        CMBlockBufferCreateWithMemoryBlock(
            allocator: nil, memoryBlock: nil, blockLength: byteCount, blockAllocator: nil,
            customBlockSource: nil, offsetToData: 0, dataLength: byteCount, flags: 0,
            blockBufferOut: &maybeBlock
        )
        guard let block = maybeBlock else { throw H3Error.exportFailed("Audio block buffer failed") }
        _ = interleaved.withUnsafeBytes { bytes in
            CMBlockBufferReplaceDataBytes(
                with: bytes.baseAddress!, blockBuffer: block, offsetIntoDestination: 0, dataLength: byteCount)
        }

        var maybeSample: CMSampleBuffer?
        CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: nil, dataBuffer: block, formatDescription: format,
            sampleCount: sampleCount, presentationTimeStamp: .zero, packetDescriptions: nil,
            sampleBufferOut: &maybeSample
        )
        guard let sampleBuffer = maybeSample else { throw H3Error.exportFailed("Audio sample buffer failed") }
        return sampleBuffer
    }
}

extension H3VideoExporter {
    /// Scale a stereo waveform so its peak sits at -3 dBFS (~0.708). Near-silent tracks
    /// (peak < -60 dBFS) are left untouched rather than amplifying noise floors.
    public static func peakNormalize(_ audio: MLXArray) -> MLXArray {
        let peak = abs(audio).max().item(Float.self)
        guard peak > 0.001 else { return audio }
        return audio * (0.708 / peak)
    }
}
