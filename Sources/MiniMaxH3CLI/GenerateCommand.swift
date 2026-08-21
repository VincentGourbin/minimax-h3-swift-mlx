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
            + "optionally anchored on keyframe images (fl2va) or on an ordered list of image, "
            + "video and audio references (ref2va)."
    )

    @Argument(help: "The text prompt.")
    var prompt: String

    @Option(name: .long, help: "fl2va: keyframe image the video starts from (sets the canvas aspect).")
    var image: String?

    @Option(name: .long, help: "fl2va: keyframe image the video ends on (alone, generates up to it).")
    var lastImage: String?

    @Option(
        name: .long, parsing: .singleValue,
        help: ArgumentHelp(
            "ref2va: a reference to condition on — an image, a video (its soundtrack comes along) "
                + "or an audio clip. Repeatable, and THE ORDER IS THE REQUEST: it numbers the "
                + "<Picture i>/<Audio j>/<Video k> labels and lays the blocks out on the shared "
                + "rotary clock. At most 9 images, 3 videos, 3 audios, 12 in total; an audio "
                + "reference cannot be used on its own.",
            valueName: "file"))
    var reference: [String] = []

    @Option(
        name: .long, parsing: .singleValue,
        help: ArgumentHelp(
            "Override the frame rate of the Nth --reference video (repeatable, positional with "
                + "the video references) when its container's metadata is wrong.",
            valueName: "fps"))
    var referenceFps: [Double] = []

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

    @Option(name: .long, help: "Step-distillation LoRA (.safetensors) folded into the transformer; 4-step Turbo adapters want -s 5, 8-step ones -s 9.")
    var turboLora: String?

    @Option(name: .long, help: "Multiplier on the folded LoRA delta (sharpness dial; 1.0 = as published).")
    var turboLoraStrength: Double = 1.0

    @Option(name: .long, help: "Block-sparse attention: fraction of key blocks kept per query block (e.g. 0.3). Approximation — validate against a full-attention run.")
    var sparseAttention: Double?

    @Flag(name: .long, help: "Verbose debug logging.")
    var debug = false

    /// Which modality a `--reference` file is, by extension. The reference implementation gets
    /// this from the container; extensions are enough here and keep the CLI from opening every
    /// file twice just to classify it.
    static func referenceKind(of url: URL) throws -> H3ReferenceKind {
        switch url.pathExtension.lowercased() {
        case "png", "jpg", "jpeg", "webp", "heic", "heif", "bmp", "tif", "tiff", "gif":
            return .image
        case "mp4", "mov", "m4v", "avi", "mkv", "webm":
            return .video
        case "wav", "mp3", "m4a", "aac", "flac", "aiff", "aif", "caf", "ogg", "opus":
            return .audio
        case let other:
            throw ValidationError(
                "--reference: cannot tell what modality '\(url.lastPathComponent)' is from its "
                    + "'.\(other)' extension. Use a known image, video or audio extension.")
        }
    }

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

        // References are decoded BEFORE the rewrite: whether a video carries a soundtrack decides
        // whether it also takes an `<Audio j>` label, and the labels are the contract between the
        // prompt and the packed sequence. Decoding is bounded by the generated duration — the
        // 24 fps resample maps output slot `j` to a source index that depends only on `j`, so
        // material past the truncation cannot change the result, and a long 4K clip is never held
        // whole.
        let requestDuration = Double(try H3Geometry.alignNumFrames(frames)) / Double(H3Constants.fps)
        var decodedReferences = [H3Reference]()
        if !reference.isEmpty {
            var videoIndex = 0
            for path in reference {
                let url = URL(fileURLWithPath: path)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw ValidationError("--reference: no file at \(url.path)")
                }
                switch try Self.referenceKind(of: url) {
                case .image:
                    decodedReferences.append(
                        .image(H3ImageReference(image: try H3KeyframeImage.load(from: url))))
                case .video:
                    let fpsOverride = videoIndex < referenceFps.count ? referenceFps[videoIndex] : nil
                    videoIndex += 1
                    decodedReferences.append(
                        .video(try await H3MediaDecoder.decodeVideo(
                            at: url, maxDuration: requestDuration, fpsOverride: fpsOverride)))
                case .audio:
                    decodedReferences.append(
                        .audio(H3AudioReference(audio: try await H3MediaDecoder.decodeAudio(
                            at: url, maxDuration: requestDuration))))
                }
            }
            try H3ReferenceNormalizer.validate(decodedReferences)
            print("References (packed order): "
                + decodedReferences.map { $0.kind.rawValue }.joined(separator: ", "))
        }

        var finalPrompt = prompt
        if enhancePrompt {
            let duration = requestDuration
            // With keyframes, the rewrite MUST see them: the text-only path would emit a T2VA
            // prompt with no `<Picture i>` reference line while the pipeline builds a vision
            // block for that same image — prompt and conditioning would describe different
            // requests, silently.
            if !reference.isEmpty {
                // Same trap as the keyframe path, one modality wider: the text-only rewrite would
                // emit a three-field T2VA prompt with no reference lines at all while the pipeline
                // builds a vision block per reference — prompt and conditioning would describe
                // different requests, silently.
                print("Enhancing prompt (Gemma 4 E4B, REF2VA)…")
                let analyzer = MultimodalContextIR()
                try await analyzer.load()
                // Labels are numbered per modality in request order, exactly as the presentation
                // numbers them — and a video that carries a soundtrack takes an `<Audio j>` label
                // as well as its `<Video k>`, because that is what the presentation emits.
                var analyses = [H3ReferenceAnalysis]()
                var pictures = 0, videos = 0, audios = 0
                for (index, path) in reference.enumerated() {
                    let url = URL(fileURLWithPath: path)
                    let decoded = decodedReferences[index]
                    if decoded.hasAudio {
                        audios += 1
                        analyses.append(H3ReferenceAnalysis(
                            label: "<Audio \(audios)>",
                            analysis: try await analyzer.describeAudio(url)))
                    }
                    switch decoded.kind {
                    case .image:
                        pictures += 1
                        analyses.append(H3ReferenceAnalysis(
                            label: "<Picture \(pictures)>",
                            analysis: try await analyzer.describeImage(url)))
                    case .video:
                        videos += 1
                        analyses.append(H3ReferenceAnalysis(
                            label: "<Video \(videos)>",
                            analysis: try await analyzer.describeVideo(url)))
                    case .audio:
                        break  // its `<Audio j>` analysis was already appended above
                    }
                }
                finalPrompt = try await analyzer.rewrite(
                    request: prompt, durationSeconds: duration, variant: .ref2va,
                    referenceAnalyses: analyses)
                await analyzer.unload()
            } else if image != nil || lastImage != nil {
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
        request.references = decodedReferences
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
        if let turboLora {
            let url = URL(fileURLWithPath: turboLora)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("--turbo-lora: no file at \(url.path)")
            }
            request.turboLoRA = url
            request.turboLoRAStrength = Float(turboLoraStrength)
            // The adapters are distilled for a specific evaluation count; running one at the 50-point
            // default wastes an hour before anyone notices the mismatch.
            let evaluations = steps - 1
            if evaluations > 12 {
                print("Note: --turbo-lora with -s \(steps) means \(evaluations) transformer "
                    + "evaluations. The published adapters are distilled for 4 (-s 5) or 8 (-s 9).")
            }
        }
        if let sparseAttention {
            guard sparseAttention > 0, sparseAttention < 1 else {
                throw ValidationError("--sparse-attention must be in (0, 1).")
            }
            request.sparseAttentionKeep = Float(sparseAttention)
        }

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
