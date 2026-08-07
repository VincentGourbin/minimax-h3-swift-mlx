// EnhanceCommand.swift - Preview the local Context-IR rewrite without generating
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import H3PromptEnhancer
import MiniMaxH3

struct EnhanceCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "enhance",
        abstract: "Rewrite a free-form prompt into H3's Context-IR format (local Gemma 4)."
    )

    @Argument(help: "The free-form prompt.")
    var prompt: String

    @Option(name: [.customShort("f"), .long], help: "Target frame count (drives timecodes; 17n+5).")
    var frames: Int = 124

    @Option(name: .long, help: "Reference image = first frame (I2VA; with --last-image: FL2VA).")
    var image: String?

    @Option(name: .long, help: "Reference image = last frame (L2VA; with --image: FL2VA).")
    var lastImage: String?

    @Option(name: .long, help: "Reference audio clip (<=30 s): soundscape/music mirror it, speech transcribed verbatim.")
    var audio: String?

    @Option(name: .long, help: "Reference video (<=60 s, ~1 fps sampling): structural inspiration.")
    var video: String?

    func run() async throws {
        let duration = Double(try H3Geometry.alignNumFrames(frames)) / Double(H3Constants.fps)
        let hasMedia = image != nil || lastImage != nil || audio != nil || video != nil

        if !hasMedia {
            let enhancer = await ContextIREnhancer()
            print("Loading Gemma 4 E4B (4-bit)…")
            let lastDecile = LockedBox(-1)
            try await enhancer.load { fraction in
                let decile = Int(fraction * 10)
                if decile > lastDecile.value {
                    lastDecile.value = decile
                    print("Downloading \(decile * 10)%")
                }
            }
            let started = Date()
            let enhanced = try await enhancer.enhance(prompt, durationSeconds: duration)
            await enhancer.unload()
            print(String(format: "— rewritten in %.1f s —\n", Date().timeIntervalSince(started)))
            print(enhanced)
            return
        }

        let variant: H3EnhanceVariant =
            image != nil && lastImage != nil ? .fl2va
            : image != nil ? .i2va
            : lastImage != nil ? .l2va
            : .t2va

        let analyzer = MultimodalContextIR()
        print("Loading Gemma 4 E4B (multimodal)…")
        try await analyzer.load()
        let started = Date()

        var imageAnalysis: String?
        if let image {
            print("Analyzing first frame…")
            imageAnalysis = try await analyzer.describeImage(URL(fileURLWithPath: image))
        }
        var lastImageAnalysis: String?
        if let lastImage {
            print("Analyzing last frame…")
            lastImageAnalysis = try await analyzer.describeImage(URL(fileURLWithPath: lastImage))
        }
        var audioAnalysis: String?
        if let audio {
            print("Analyzing audio…")
            audioAnalysis = try await analyzer.describeAudio(URL(fileURLWithPath: audio))
        }
        var videoAnalysis: String?
        if let video {
            print("Analyzing video…")
            videoAnalysis = try await analyzer.describeVideo(URL(fileURLWithPath: video))
        }

        print("Rewriting (\(variant.rawValue.uppercased()))…")
        let enhanced = try await analyzer.rewrite(
            request: prompt, durationSeconds: duration, variant: variant,
            imageAnalysis: imageAnalysis, lastImageAnalysis: lastImageAnalysis,
            audioAnalysis: audioAnalysis, videoAnalysis: videoAnalysis
        )
        await analyzer.unload()
        print(String(format: "— rewritten in %.1f s —\n", Date().timeIntervalSince(started)))
        print(enhanced)
    }
}


/// Minimal thread-safe box for progress throttling across Sendable callbacks.
final class LockedBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Int
    init(_ value: Int) { stored = value }
    var value: Int {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}
