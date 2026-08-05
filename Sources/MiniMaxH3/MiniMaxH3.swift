// MiniMaxH3.swift - Umbrella: version, errors, debug logging
// Copyright 2026 Vincent Gourbin

import Foundation
@_exported import MLXProfiler

/// Cross-project profiling convention (same as LTXVideoProfiler in ltx-video-swift-mlx).
public typealias H3Profiler = MLXProfiler

/// MiniMax-H3 for Swift MLX — joint video + stereo audio generation on Apple Silicon.
public enum MiniMaxH3 {
    public static let version = "0.1.0"
    public static let name = "minimax-h3-swift-mlx"
}

// MARK: - Errors

public enum H3Error: Error, LocalizedError, Sendable {
    case modelNotFound(String)
    case weightLoadingFailed(String)
    case invalidConfiguration(String)
    case invalidInput(String)
    case generationFailed(String)
    case exportFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let detail): return "Model not found: \(detail)"
        case .weightLoadingFailed(let detail): return "Weight loading failed: \(detail)"
        case .invalidConfiguration(let detail): return "Invalid configuration: \(detail)"
        case .invalidInput(let detail): return "Invalid input: \(detail)"
        case .generationFailed(let detail): return "Generation failed: \(detail)"
        case .exportFailed(let detail): return "Export failed: \(detail)"
        }
    }
}

// MARK: - Debug logging

public enum H3Debug {
    nonisolated(unsafe) public static var isEnabled = false

    public static func log(_ message: @autoclosure () -> String) {
        if isEnabled {
            print("[H3] \(message())")
            fflush(stdout)
        }
    }
}
