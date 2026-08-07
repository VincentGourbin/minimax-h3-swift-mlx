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

    func run() async throws {
        let duration = Double(try H3Geometry.alignNumFrames(frames)) / Double(H3Constants.fps)
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
