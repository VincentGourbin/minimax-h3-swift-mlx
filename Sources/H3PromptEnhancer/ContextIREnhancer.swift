// ContextIREnhancer.swift - Local substitute for MiniMax's hosted H3-Context-IR
// Copyright 2026 Vincent Gourbin
//
// Rewrites a free-form prompt into the structured Context-IR format H3-Base was trained on
// (integrated_multimodal_description / overall_soundscape / non_diegetic_music, [Shot N]
// timecodes), using Gemma 4 locally via gemma-4-swift-mlx. The system prompt condenses
// MiniMax's official VIDEO_PROMPT_WRITING_GUIDE_base_en.md.

import Foundation
import Gemma4Swift
import MiniMaxH3

public enum ContextIREnhancerError: Error, LocalizedError {
    case malformedOutput(String)

    public var errorDescription: String? {
        switch self {
        case .malformedOutput(let raw):
            return "The rewriter did not produce a valid Context-IR prompt. Raw output:\n\(raw)"
        }
    }
}

@MainActor
public final class ContextIREnhancer {
    private let pipeline = Gemma4Pipeline()
    private var loaded = false

    public init() {}

    /// Load the rewriter model (Gemma 4 E4B 4-bit, text-only ≈ 5 GB), downloading if needed.
    public func load(progress: (@Sendable (Double) -> Void)? = nil) async throws {
        guard !loaded else { return }
        // Text-only: the rewriter needs no vision tower (~2 GB and ~4.6x throughput saved;
        // gemma-4-swift-mlx >= 1.0.0 fixed the E4B text-only load path).
        try await pipeline.load(.e4b4bit, multimodal: false, downloadIfNeeded: true) { p in
            progress?(p.fraction)
        }
        loaded = true
    }

    /// Free the model (Gemma is only needed for a few seconds per request).
    public func unload() {
        pipeline.unload()
        loaded = false
    }

    /// Rewrite `prompt` into the Context-IR format for a video of `durationSeconds`.
    public func enhance(_ prompt: String, durationSeconds: Double) async throws -> String {
        let user = """
            Target video duration: \(String(format: "%.2f", durationSeconds)) seconds.
            User request: \(prompt)
            """
        let raw = try await pipeline.chat(
            prompt: user,
            systemPrompt: Self.systemPrompt,
            temperature: 0.3,
            maxTokens: 900
        )
        let cleaned = Self.stripDecorations(raw)
        guard cleaned.contains("integrated_multimodal_description:"),
              cleaned.contains("overall_soundscape:"),
              cleaned.contains("non_diegetic_music:") else {
            throw ContextIREnhancerError.malformedOutput(raw)
        }
        return cleaned
    }

    /// Remove Gemma channel markers (thinking blocks) and code fences from the raw completion.
    nonisolated static func stripDecorations(_ raw: String) -> String {
        var text = raw
        // Keep only the content after the last response-channel marker, if any are present.
        if let range = text.range(of: "<|channel|>response", options: .backwards) {
            text = String(text[range.upperBound...])
        } else if let range = text.range(of: "<|channel|>thought", options: .backwards) {
            // A thought block with no response marker: drop everything before the blank line
            // that usually separates thought from answer, else keep as-is and let validation act.
            if let separator = text[range.upperBound...].range(of: "\n\n") {
                text = String(text[separator.upperBound...])
            }
        }
        text = text.replacingOccurrences(of: "```text", with: "")
        text = text.replacingOccurrences(of: "```", with: "")
        // Trailing turn-marker fragments occasionally leak through the token filter.
        text = text.replacingOccurrences(
            of: "<[|]?(turn|end_of_turn|eot)[|]?>", with: "", options: .regularExpression)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - System prompt (condensed from MiniMax's official prompt-writing guide)

    nonisolated static let systemPrompt = """
        You are H3-Context-IR, the prompt-structuring stage of the MiniMax-H3 video+audio \
        generator. Rewrite the user's request into EXACTLY this format, in English, and output \
        NOTHING else — no commentary, no markdown fences, no reasoning:

        integrated_multimodal_description: [Shot 1] <style>, <initial composition and subjects>, \
        <actions, camera motion, dialogue and diegetic sound along the timeline> [Shot 2] At \
        <MM:SS.mmm>, the camera cuts to <...>

        overall_soundscape: <1-4 sentences: ambient sound, physical action sounds, non-verbal \
        human sounds across the whole video>

        non_diegetic_music: <1-3 sentences about background score the characters cannot hear: \
        instrumentation, tempo, rhythm, dynamics — no mood words>, or N/A if none fits the request

        Rules:
        - Start [Shot 1] with an overall style (Cinematic, live-action, 2D-animated, 3D CG, \
        claymation, watercolor, vintage film...) and the initial composition.
        - No timestamp on Shot 1. Later shots start with a strictly increasing cut time within \
        the target duration, e.g. "[Shot 2] At 00:03.500, the camera cuts to". Prefer a single \
        shot for short videos; cut only when it adds new information.
        - Camera motion is written as natural action inside the shot: motion type (Push In, Pull \
        Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, \
        Tracking Shot, Static Shot, Zoom In/Out, POV, Shake Slightly/Strongly, Roll \
        Clockwise/Counterclockwise) + optional amplitude ("with small/large amplitude") + \
        optional speed ("at slow/fast speed").
        - Speakers get stable IDs: "the young woman with a quiet, breathy voice (S1) says: \
        <d>[English] ...</d>". Inside <d> put only the language tag and the exact spoken words. \
        CRITICAL: user-provided dialogue is copied VERBATIM in its ORIGINAL language, never \
        translated — a French line stays French with the [French] tag: the co-pilot (S1) \
        shouts: <d>[French] virage serré à droite!</d>. Only the surrounding description is \
        in English. Characters who never vocalize get no ID. For voiceover use exactly "says \
        in an off-screen voiceover" and state that the on-screen lips remain closed.
        - Visible on-screen text goes in double quotes, verbatim, untranslated.
        - Every detail must be visible or audible; keep subjects, clothing, colors and spatial \
        relations consistent across shots.
        - overall_soundscape must NOT repeat dialogue or music; name each sound source and its \
        character (loud, soft, close-miked, distant...). Loudness words directly drive the mix.
        - Use N/A for non_diegetic_music when the user asks for none or the scene calls for none.
        - Timecodes and pacing must fit the target duration given by the user.

        Example (user request: "a baker opens his bakery at dawn", duration 8.00 s):
        integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot \
        frames a baker opening the shutters of a small street bakery before sunrise. The camera \
        pushes in with small amplitude at slow speed as the middle-aged baker with a calm, \
        slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: \
        <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to \
        a close-up of steam rising from the sliced bread while the baker's final words carry \
        over from the previous shot.

        overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink \
        softly inside the bakery. The doorbell rings once, followed by light footsteps and the \
        crisp sound of bread being sliced.

        non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by \
        sparse upright-bass notes and a gentle fade at the end.
        """
}
