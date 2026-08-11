// GenerateCommand.swift - t2va generation subcommand
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import H3PromptEnhancer
import MiniMaxH3
import MLX

struct GenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate a video with synchronized stereo audio from a text prompt (t2va), "
            + "optionally anchored on keyframe images (fl2va)."
    )

    @Argument(help: "The text prompt.")
    var prompt: String

    @Option(name: .long, help: "fl2va: keyframe image the video starts from (sets the canvas aspect).")
    var image: String?

    @Option(name: .long, help: "fl2va: keyframe image the video ends on (alone, generates up to it).")
    var lastImage: String?

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

    @Option(name: .long, help: "Transformer quantization: none | qint8 | qint6 | int4.")
    var transformerQuant: String = "none"

    @Option(name: .long, help: "Text encoder quantization: none | qint8 | qint6 | int4.")
    var textEncoderQuant: String = "none"

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    @Flag(name: .long, help: "Allow durations under 5 s (smoke tests).")
    var allowShortVideo = false

    @Flag(name: .long, help: "Profile phases/steps/memory; prints a report and writes a Perfetto trace.")
    var profile = false

    @Flag(name: .long, help: "Peak-normalize the audio track to -3 dBFS at mux time.")
    var normalizeAudio = false

    @Flag(name: .long, help: "Rewrite the prompt into Context-IR format with local Gemma 4 first.")
    var enhancePrompt = false

    @Flag(name: .long, help: "Graph-compile the transformer blocks (fused kernels, same math).")
    var compileBlocks = false

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
            profilingSession.metadata["quant"] = "\(transformerQuant)/\(textEncoderQuant)"
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

        var finalPrompt = prompt
        if enhancePrompt {
            let duration = Double(try H3Geometry.alignNumFrames(frames)) / Double(H3Constants.fps)
            // With keyframes, the rewrite MUST see them: the text-only path would emit a T2VA
            // prompt with no `<Picture i>` reference line while the pipeline builds a vision
            // block for that same image — prompt and conditioning would describe different
            // requests, silently.
            if image != nil || lastImage != nil {
                let variant: H3EnhanceVariant =
                    image != nil && lastImage != nil ? .fl2va : image != nil ? .i2va : .l2va
                print("Enhancing prompt (Gemma 4 E4B, \(variant.rawValue.uppercased()))…")
                let analyzer = MultimodalContextIR()
                try await analyzer.load()
                var imageAnalysis: String?
                if let image {
                    imageAnalysis = try await analyzer.describeImage(URL(fileURLWithPath: image))
                }
                var lastImageAnalysis: String?
                if let lastImage {
                    lastImageAnalysis = try await analyzer.describeImage(
                        URL(fileURLWithPath: lastImage))
                }
                finalPrompt = try await analyzer.rewrite(
                    request: prompt, durationSeconds: duration, variant: variant,
                    imageAnalysis: imageAnalysis, lastImageAnalysis: lastImageAnalysis,
                    audioAnalysis: nil, videoAnalysis: nil)
                await analyzer.unload()
            } else {
                print("Enhancing prompt (Gemma 4 E4B)…")
                let enhancer = await ContextIREnhancer()
                try await enhancer.load(progress: nil)
                finalPrompt = try await enhancer.enhance(prompt, durationSeconds: duration)
                await enhancer.unload()
            }
            print("— Context-IR prompt —\n\(finalPrompt)\n———")
            let promptURL = URL(fileURLWithPath: output).deletingPathExtension()
                .appendingPathExtension("prompt.txt")
            try? finalPrompt.write(to: promptURL, atomically: true, encoding: .utf8)
        }

        var request = H3GenerationRequest(prompt: finalPrompt)
        if let image {
            request.image = try H3KeyframeImage.load(from: URL(fileURLWithPath: image))
        }
        if let lastImage {
            request.lastImage = try H3KeyframeImage.load(from: URL(fileURLWithPath: lastImage))
        }
        request.height = height
        request.width = width
        request.numFrames = frames
        request.numInferenceSteps = steps
        request.seed = seed
        request.flowShift = flowShift
        request.audioFlowShift = audioFlowShift
        request.allowShortVideo = allowShortVideo
        guard let transformerQuantization = H3Quantization(rawValue: transformerQuant),
              let textEncoderQuantization = H3Quantization(rawValue: textEncoderQuant) else {
            throw ValidationError("Quantization must be one of: "
                + H3Quantization.allCases.map(\.rawValue).joined(separator: ", "))
        }
        request.transformerQuantization = transformerQuantization
        request.textEncoderQuantization = textEncoderQuantization
        request.compileBlocks = compileBlocks

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
