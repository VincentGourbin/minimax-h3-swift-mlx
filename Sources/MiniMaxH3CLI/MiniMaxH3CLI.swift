// MiniMaxH3CLI.swift - CLI entry point
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MiniMaxH3

@main
struct MiniMaxH3CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "minimax-h3",
        abstract: "MiniMax-H3 joint video + stereo audio generation on Apple Silicon (MLX).",
        version: MiniMaxH3.version,
        subcommands: [GenerateCommand.self, EnhanceCommand.self, MuxCommand.self, ExportQuantizedCommand.self, SmokeCommand.self, ParityCommand.self, InfoCommand.self],
        defaultSubcommand: InfoCommand.self
    )
}

struct InfoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show version and model directory status."
    )

    @Option(name: .long, help: "Model directory (diffusers layout).")
    var modelsDir: String = defaultModelsDirectory()

    func run() async throws {
        print("\(MiniMaxH3.name) \(MiniMaxH3.version)")
        let url = URL(fileURLWithPath: modelsDir)
        for component in ["transformer", "text_encoder", "vae", "audio_vae", "tokenizer", "scheduler"] {
            let path = url.appendingPathComponent(component)
            let exists = FileManager.default.fileExists(atPath: path.path)
            print("  \(exists ? "✓" : "✗") \(component)")
        }
    }
}
