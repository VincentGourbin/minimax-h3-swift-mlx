// MuxCommand.swift - Rebuild an MP4 from a saved raw generation result
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX

struct MuxCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mux",
        abstract: "Mux a saved raw result (<name>.raw.safetensors) into an MP4."
    )

    @Argument(help: "Path to the .raw.safetensors file written by `generate`.")
    var rawPath: String

    @Option(name: .shortAndLong, help: "Output MP4 path.")
    var output: String = "output.mp4"

    @Flag(name: .long, help: "Peak-normalize the audio track to -3 dBFS at mux time.")
    var normalizeAudio = false

    func run() async throws {
        H3Debug.isEnabled = true
        let arrays = try loadArrays(url: URL(fileURLWithPath: rawPath))
        guard let frames = arrays["frames"], let audio = arrays["audio"] else {
            throw ValidationError("Raw file must contain 'frames' and 'audio'.")
        }
        let fps = arrays["fps"].map { Int($0.item(Int32.self)) } ?? 24
        let sampleRate = arrays["sample_rate"].map { Int($0.item(Int32.self)) } ?? 32_000

        let outputURL = URL(fileURLWithPath: output)
        let audioTrack = normalizeAudio ? H3VideoExporter.peakNormalize(audio) : audio
        try await H3VideoExporter.export(
            frames: frames, audio: audioTrack, fps: fps, sampleRate: sampleRate, to: outputURL)
        print("Wrote \(outputURL.path)")
    }
}
