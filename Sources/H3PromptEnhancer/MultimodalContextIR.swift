// MultimodalContextIR.swift - Media-aware Context-IR rewriting (image / audio / video inputs)
// Copyright 2026 Vincent Gourbin
//
// Applies the H3 cookbook recipes to Gemma 4's multimodal capabilities: analyze reference media
// with the E4B towers (vision 280 tok/image, audio <=30 s, video ~1 fps <=32 frames), then write
// the official I2VA / FL2VA / L2VA / T2VA prompt formats around the analyses. The generation
// path replicates gemma4-cli's canonical `describe` flow (chat template -> special-token
// expansion -> pending injection -> prefill/decode with the thinking filter), since
// Gemma4Pipeline keeps its container private and only supports single-image chat.

import Foundation
import Gemma4Swift
import MiniMaxH3
import MLX
import MLXLMCommon
import MLXRandom

public enum H3EnhanceVariant: String, Sendable {
    case t2va, i2va, fl2va, l2va, ref2va

    /// The section the rewrite must open with. `ref2va` takes six sections of its own instead of
    /// the three-field Context-IR the other four modes share.
    var leadSection: String {
        self == .ref2va ? "subject_definitions:" : "integrated_multimodal_description:"
    }
}

/// One `ref2va` reference as the rewriter sees it: the label MiniMax-H3 will resolve it under,
/// and what the analyzer saw in it.
public struct H3ReferenceAnalysis: Sendable {
    /// `"<Picture 1>"`, `"<Video 2>"`, `"<Audio 1>"` — numbered per modality in request order,
    /// exactly as the presentation numbers them.
    public var label: String
    public var analysis: String

    public init(label: String, analysis: String) {
        self.label = label
        self.analysis = analysis
    }
}

public actor MultimodalContextIR {
    public enum Failure: Error, LocalizedError {
        case modelMissing(String)
        case malformedOutput(String)

        public var errorDescription: String? {
            switch self {
            case .modelMissing(let path):
                return "Gemma 4 E4B not found at \(path) — run `minimax-h3 enhance \"...\"` once "
                    + "(text-only) to download it."
            case .malformedOutput(let raw):
                return "The rewriter did not produce a valid Context-IR prompt. Raw output:\n\(raw)"
            }
        }
    }

    private var container: ModelContainer?

    public init() {}

    public func load() async throws {
        guard container == nil else { return }
        let modelDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit")
        guard FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("config.json").path) else {
            throw Failure.modelMissing(modelDirectory.path)
        }
        await Gemma4Registration.register(multimodal: true)
        container = try await loadModelContainer(
            from: modelDirectory, using: Gemma4TokenizerLoader())
    }

    public func unload() {
        container = nil
        MLX.GPU.clearCache()
    }

    // MARK: - Media analysis (cookbook-style descriptions)

    /// Describe a keyframe the way the reference guide wants Shot 1 anchored: style, subjects,
    /// wardrobe/colors, composition, environment, lighting.
    public func describeImage(_ url: URL) async throws -> String {
        let pixels = try Gemma4ImageProcessor.processImage(url: url)
        return try await generate(
            content: "<|image|>\nDescribe this image for film production, in English, one dense "
                + "paragraph: overall visual style, every subject with appearance/clothing/colors, "
                + "their positions and spatial relations, the environment and key props, lighting "
                + "and atmosphere, and the exact composition/framing. Only what is visible.",
            pixelValues: pixels, imageCount: 1,
            maxTokens: 400
        )
    }

    /// Describe a short audio clip per the cookbook's audio-reference recipe: sources, character,
    /// rhythm/tempo, and a verbatim transcript of any speech or lyrics in the original language.
    public func describeAudio(_ url: URL) async throws -> String {
        let features = try await Gemma4AudioProcessor.processAudio(url: url)
        return try await generate(
            content: "<|audio|>\nAnalyze this audio, in English: name each sound source and its "
                + "character (loud/soft, close/distant, texture), the rhythm and tempo of any "
                + "music with its instrumentation, and transcribe any speech or lyrics VERBATIM "
                + "in their original language, marked as: transcript: \"...\". Only what is audible.",
            audioFeatures: features,
            maxTokens: 400
        )
    }

    /// Describe a short video per the reference-generation recipe: shots, subjects, camera
    /// motion, pacing, and audible events.
    public func describeVideo(_ url: URL) async throws -> String {
        let frames = try await Gemma4VideoProcessor.processVideo(url: url)
        return try await generate(
            content: "Describe this video for film production, in English: the shots in order "
                + "with what changes at each cut, subjects and their actions, camera motion "
                + "(type, amplitude, speed), pacing, environment and lighting.",
            videoFrames: frames,
            maxTokens: 450
        )
    }

    // MARK: - Context-IR rewrite

    /// Rewrite a user request into the official Context-IR format for `variant`, weaving in the
    /// media analyses (image-anchored alignment lines for i2va/fl2va/l2va).
    public func rewrite(
        request: String,
        durationSeconds: Double,
        variant: H3EnhanceVariant,
        imageAnalysis: String? = nil,
        lastImageAnalysis: String? = nil,
        audioAnalysis: String? = nil,
        videoAnalysis: String? = nil,
        referenceAnalyses: [H3ReferenceAnalysis] = []
    ) async throws -> String {
        var sections = [String]()
        sections.append("Target video duration: \(String(format: "%.2f", durationSeconds)) seconds.")
        sections.append("Task variant: \(variant.rawValue.uppercased()).")
        if let imageAnalysis {
            sections.append("Analysis of <Picture 1> (the reference frame):\n\(imageAnalysis)")
        }
        if let lastImageAnalysis {
            let label = imageAnalysis == nil ? "<Picture 1>" : "<Picture 2>"
            sections.append("Analysis of \(label) (the LAST frame):\n\(lastImageAnalysis)")
        }
        if let audioAnalysis {
            sections.append(
                "Analysis of the reference audio (mirror its content in overall_soundscape / "
                    + "non_diegetic_music; reuse any transcript verbatim inside <d>):\n\(audioAnalysis)")
        }
        if let videoAnalysis {
            sections.append(
                "Analysis of the reference video (use it as structural inspiration — shots, "
                    + "camera, pacing):\n\(videoAnalysis)")
        }
        for reference in referenceAnalyses {
            sections.append("Analysis of \(reference.label):\n\(reference.analysis)")
        }
        if !referenceAnalyses.isEmpty {
            // The labels are the contract between the prompt and the packed sequence: the
            // presentation numbers them per modality in request order, and a label the rewrite
            // invents (or renumbers) points at conditioning that is not there.
            sections.append(
                "Reference labels available, and the ONLY ones you may use: "
                    + referenceAnalyses.map(\.label).joined(separator: ", ") + ".")
        }
        sections.append("User request: \(request)")

        let raw = try await generate(
            content: sections.joined(separator: "\n\n"),
            systemPrompt: Self.variantSystemPrompt(variant, durationSeconds: durationSeconds),
            maxTokens: 900
        )
        var cleaned = ContextIREnhancer.stripDecorations(raw)
        guard let core = cleaned.range(of: variant.leadSection) else {
            throw Failure.malformedOutput(raw)
        }
        // The alignment line is a fixed contract — impose it deterministically instead of
        // trusting the model to reproduce it verbatim.
        cleaned = String(cleaned[core.lowerBound...])
        if let alignment = Self.alignmentLine(variant, durationSeconds: durationSeconds) {
            cleaned = alignment + "\n\n" + cleaned
        }
        return cleaned
    }

    /// The exact instruction line the cookbook mandates for keyframe variants.
    static func alignmentLine(_ variant: H3EnhanceVariant, durationSeconds: Double) -> String? {
        let duration = String(format: "%.2f", durationSeconds)
        switch variant {
        case .t2va:
            return nil
        case .i2va:
            return "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                + "(from [Shot 1]) is fully referenced."
        case .fl2va:
            return "How the reference pictures align with the target video — Picture 1 (from "
                + "Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 "
                + "(from Shot 1) aligns with the \(duration)-second mark of the target video."
        case .l2va:
            return "How the reference pictures align with the target video — <Picture 1> (from "
                + "[Shot 1]) aligns with the \(duration)-second mark of the target video."
        case .ref2va:
            // No keyframe, so no alignment line: `subject_definitions` opens the rewrite and the
            // reference relationships are stated in `retention_analysis` instead.
            return nil
        }
    }

    static func variantSystemPrompt(_ variant: H3EnhanceVariant, durationSeconds: Double) -> String {
        let duration = String(format: "%.2f", durationSeconds)
        var prompt = ContextIREnhancer.systemPrompt
        switch variant {
        case .t2va:
            break
        case .i2va:
            prompt += """


                I2VA variant: the FIRST LINE of your output must be exactly:
                For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
                Then one blank line, then the three core fields. [Shot 1] must begin from the \
                provided <Picture 1> analysis — preserve its style, subjects, appearance, \
                clothing, colors and composition, then develop the action forward: first-frame \
                anchor, action onset, continuous development, result.
                """
        case .fl2va:
            prompt += """


                FL2VA variant: the FIRST LINE of your output must be exactly:
                How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the \(duration)-second mark of the target video.
                Then one blank line, then the three core fields. Prefer a single continuous \
                shot: start from the <Picture 1> state, describe the observable path of motion \
                and change, and land exactly on the <Picture 2> state at the end.
                """
        case .l2va:
            prompt += """


                L2VA variant: the FIRST LINE of your output must be exactly:
                How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the \(duration)-second mark of the target video.
                Then one blank line, then the three core fields. Infer a plausible earlier \
                state, then converge: preceding state, explicit transition path, gradual \
                convergence, last-frame landing on the <Picture 1> state.
                """
        case .ref2va:
            // Full-reference mode takes SIX sections of its own, in this order — the format
            // MiniMax documents in `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`. It replaces the
            // three-field Context-IR entirely rather than extending it.
            prompt = """
                You rewrite a user's request into MiniMax-H3's full-reference (Ref2VA) format.
                Output ONLY the rewrite, in English, with exactly these six sections in this \
                order, each on its own line and separated by a blank line:

                subject_definitions:
                summary:
                retention_analysis:
                detailed_description:
                overall_soundscape:
                non_diegetic_music:

                subject_definitions: one line per tracked item. Introduce `<Subject N>` for each \
                person, animal, environment or object that has to stay consistent, and say which \
                reference label it comes from — "<Subject 1> is the ... in <Picture 1>, featuring \
                ...". An `<Audio N>` that is only a voice or music reference gets its own line \
                stating what it provides. A reference label cited only to source another item \
                needs no line of its own.

                summary: one short paragraph opening with a square-bracketed task-type prefix \
                built from the roles the references actually play — `reference generation`, \
                `video editing`, `video continuation`, `audio reuse`, `audio reference`, \
                `keyframe completion` — e.g. "[reference generation + audio reference]".

                retention_analysis: one line per reference label, naming the shots it appears in \
                and its relationship — `fully_preserved`, `partially_preserved`, `transferred`, \
                `reused` or `reference` — followed by " - " and what is retained. Never write a \
                speaker id here.

                detailed_description: the playback-order description, 350-500 words, in \
                `[Shot N]` blocks with `At 00:MM.mmm` timecodes on every shot after the first. \
                Establish composition, appearance, environment, lighting, action, camera move \
                and sound in each shot, and cite the reference labels where their content \
                actually appears. Assign a speaker id `(Sx)` at the first vocal event of each \
                distinct voice, in order, and reuse it at every later line by that voice. Write \
                dialogue and lyrics ONLY here, inside `<d>[Language] ...</d>`, in their original \
                language.

                overall_soundscape: ambience and physical sound across the whole video, with \
                explicit loudness words. No dialogue text.

                non_diegetic_music: audience-only score — instrumentation, tempo, development — \
                or `N/A` when there is none.

                A duration of \(duration) seconds is the target; the described timeline must fit \
                it. Use ONLY the reference labels you were given, keep each label's meaning \
                identical across all six sections, and never invent a label. Describe concretely: \
                no plot summaries, no unresolved labels, no abstract adjectives standing in for \
                description.
                """
        }
        return prompt
    }

    // MARK: - Canonical generation path (mirrors gemma4-cli describe)

    private func generate(
        content: String,
        systemPrompt: String? = nil,
        pixelValues: MLXArray? = nil,
        imageCount: Int = 0,
        audioFeatures: Gemma4AudioProcessor.AudioFeatures? = nil,
        videoFrames: Gemma4VideoProcessor.VideoFrames? = nil,
        maxTokens: Int,
        temperature: Float = 0.3
    ) async throws -> String {
        guard let container else { throw Failure.modelMissing("(not loaded)") }

        var body = content
        if let videoFrames {
            let placeholders = (0..<videoFrames.frameCount).map { index in
                "\(Gemma4VideoProcessor.formatTimestamp(videoFrames.timestamps[index]))\n<|video|>"
            }
            body = placeholders.joined(separator: "\n") + "\n" + content
        }

        var builtMessages = [[String: String]]()
        if let systemPrompt { builtMessages.append(["role": "system", "content": systemPrompt]) }
        builtMessages.append(["role": "user", "content": body])
        let messages = builtMessages

        let audioTokens = audioFeatures?.numTokens ?? 0
        let softPerFrame = videoFrames?.softTokensPerFrame ?? 0
        let rawIds: [Int] = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(messages: messages)
        }

        // Expand each media placeholder id into begin + N soft tokens + end.
        let imageId = Int(Gemma4Processor.imageTokenId)
        let videoId = Int(Gemma4Processor.videoTokenId)
        let audioId = Int(Gemma4Processor.audioTokenId)
        var ids = [Int]()
        ids.reserveCapacity(rawIds.count + imageCount * 282 + audioTokens + 2
            + (videoFrames?.frameCount ?? 0) * (softPerFrame + 2))
        for tokenId in rawIds {
            switch tokenId {
            case imageId:
                ids.append(Int(Gemma4Processor.boiTokenId))
                ids.append(contentsOf: repeatElement(imageId, count: 280))
                ids.append(Int(Gemma4Processor.eoiTokenId))
            case videoId:
                ids.append(Int(Gemma4Processor.boiTokenId))
                ids.append(contentsOf: repeatElement(videoId, count: softPerFrame))
                ids.append(Int(Gemma4Processor.eoiTokenId))
            case audioId:
                ids.append(Int(Gemma4Processor.boaTokenId))
                ids.append(contentsOf: repeatElement(audioId, count: audioTokens))
                ids.append(Int(Gemma4Processor.eoaTokenId))
            default:
                ids.append(tokenId)
            }
        }

        nonisolated(unsafe) let inputIds = MLXArray(ids.map(Int32.init))
        nonisolated(unsafe) let capturedPixels = pixelValues
        nonisolated(unsafe) let capturedAudio = audioFeatures
        nonisolated(unsafe) let capturedVideo = videoFrames
        let tokenFilter = Gemma4TokenFilter(mode: .disabled)

        let generated: [Int] = try await container.perform { context in
            if let model = context.model as? Gemma4MultimodalLLMModel {
                model.pendingPixelValues = capturedPixels
                if let video = capturedVideo {
                    model.pendingVideoFrames = video.pixelValues
                    model.pendingVideoSoftTokensPerFrame = video.softTokensPerFrame
                }
                if let audio = capturedAudio {
                    model.pendingAudioFeatures = audio.features
                    model.pendingAudioMask = audio.mask
                }
            }

            let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: 0.95)
            let cache = context.model.newCache(parameters: params)
            let prefill = context.model(inputIds.reshaped(1, -1), cache: cache)
            var next = argMax(prefill[0..., prefill.dim(1) - 1, 0...], axis: -1).item(Int32.self)

            var tokens = [Int]()
            var visible = 0
            for _ in 0..<(maxTokens * 3) {  // thinking-token budget on top of the visible budget
                tokens.append(Int(next))
                let piece = context.tokenizer.decode(tokenIds: [Int(next)])
                if !tokenFilter.process(tokenId: next, text: piece).isEmpty { visible += 1 }
                if tokenFilter.isEOS(next) { break }
                if visible >= maxTokens { break }

                let output = context.model(MLXArray([next]).reshaped(1, 1), cache: cache)
                if temperature <= 0.01 {
                    next = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)
                } else {
                    let probabilities = softmax(output[0..., 0, 0...] / temperature, axis: -1)
                    next = MLXRandom.categorical(log(probabilities)).item(Int32.self)
                }
            }
            return tokens
        }

        // Re-run the filter over the whole decode to keep only the response channel.
        let replay = Gemma4TokenFilter(mode: .disabled)
        var text = ""
        for tokenId in generated {
            let piece: String = await container.perform { context in
                context.tokenizer.decode(tokenIds: [tokenId])
            }
            text += replay.process(tokenId: Int32(tokenId), text: piece)
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
