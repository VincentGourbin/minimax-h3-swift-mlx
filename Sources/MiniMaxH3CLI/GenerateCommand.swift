// GenerateCommand.swift - t2va generation subcommand
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX

struct GenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate a video with synchronized stereo audio from a text prompt (t2va)."
    )

    @Argument(help: "The text prompt.")
    var prompt: String

    @Option(name: .shortAndLong, help: "Output MP4 path.")
    var output: String = "output.mp4"

    @Option(name: [.customShort("W"), .long], help: "Canvas width (multiple of 32).")
    var width: Int?

    @Option(name: [.customShort("H"), .long], help: "Canvas height (multiple of 32).")
    var height: Int?

    @Option(name: [.customShort("f"), .long], help: "Number of frames (snapped up to 17n+5; 24 fps).")
    var frames: Int = 124

    @Option(name: [.customShort("s"), .long], help: "Sigma grid points (steps-1 transformer evals).")
    var steps: Int = 50

    @Option(name: .long, help: "Random seed.")
    var seed: UInt64 = 0

    @Option(name: .long, help: "Video flow shift.")
    var flowShift: Float = 12.0

    @Option(name: .long, help: "Audio flow shift.")
    var audioFlowShift: Float = 3.0

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    @Flag(name: .long, help: "Allow durations under 5 s (smoke tests).")
    var allowShortVideo = false

    @Flag(name: .long, help: "Profile phases/steps/memory; prints a report and writes a Perfetto trace.")
    var profile = false

    @Flag(name: .long, help: "Peak-normalize the audio track to -3 dBFS at mux time.")
    var normalizeAudio = false

    @Flag(name: .long, help: "Verbose debug logging.")
    var debug = false

    func run() async throws {
        H3Debug.isEnabled = debug

        var session: ProfilingSession?
        if profile {
            let profilingSession = ProfilingSession(config: ProfilingConfig(trackMemory: true))
            profilingSession.title = "MINIMAX-H3 T2VA"
            profilingSession.metadata["canvas"] = "\(width.map(String.init) ?? "auto")x\(height.map(String.init) ?? "auto")"
            profilingSession.metadata["frames"] = "\(frames)"
            profilingSession.metadata["steps"] = "\(steps)"
            profilingSession.metadata["seed"] = "\(seed)"
            H3Profiler.shared.enable()
            H3Profiler.shared.activeSession = profilingSession
            session = profilingSession
        }
        defer {
            if profile {
                H3Profiler.shared.activeSession = nil
                H3Profiler.shared.disable()
            }
        }

        var request = H3GenerationRequest(prompt: prompt)
        request.height = height
        request.width = width
        request.numFrames = frames
        request.numInferenceSteps = steps
        request.seed = seed
        request.flowShift = flowShift
        request.audioFlowShift = audioFlowShift
        request.allowShortVideo = allowShortVideo

        let pipeline = H3Pipeline(modelDirectory: URL(fileURLWithPath: modelsDir))
        pipeline.progressHandler = { phase, step, total in
            if total > 0 {
                print("\r\(phase) \(step)/\(total)", terminator: step == total ? "\n" : "")
                fflush(stdout)
            } else {
                print(phase)
            }
        }

        let start = Date()
        let result = try await pipeline.generate(request)
        print(String(format: "Generation finished in %.1f s", Date().timeIntervalSince(start)))

        let outputURL = URL(fileURLWithPath: output)
        // Insurance: persist the raw result before muxing, so an export failure never costs a
        // denoising run. `minimax-h3 mux <raw> -o out.mp4` rebuilds the MP4 from this file.
        let rawURL = outputURL.deletingPathExtension().appendingPathExtension("raw.safetensors")
        try MLX.save(
            arrays: [
                "frames": result.videoFrames,
                "audio": result.audioWaveform,
                "fps": MLXArray(Int32(result.fps)),
                "sample_rate": MLXArray(Int32(result.sampleRate)),
            ],
            url: rawURL
        )
        print("Raw result saved to \(rawURL.path)")

        var audioTrack = result.audioWaveform
        if normalizeAudio {
            audioTrack = H3VideoExporter.peakNormalize(audioTrack)
        }
        H3Profiler.shared.start("Video Write")
        try await H3VideoExporter.export(
            frames: result.videoFrames,
            audio: audioTrack,
            fps: result.fps,
            sampleRate: result.sampleRate,
            to: outputURL
        )
        H3Profiler.shared.end("Video Write")
        print("Wrote \(outputURL.path)")

        if let session {
            print(session.generateReport())
            let traceURL = outputURL.deletingPathExtension().appendingPathExtension("trace.json")
            try ChromeTraceExporter.export(session: session).write(to: traceURL)
            print("Perfetto trace: \(traceURL.path)")
        }
    }
}
