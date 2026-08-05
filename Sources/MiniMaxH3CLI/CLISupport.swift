// CLISupport.swift - Shared CLI helpers
// Copyright 2026 Vincent Gourbin

import Foundation

/// Default model directory: `$H3_MODELS_DIR` if set, else `~/Library/Caches/minimax-h3-mlx`.
/// Every command also accepts an explicit `--models-dir`.
func defaultModelsDirectory() -> String {
    if let fromEnvironment = ProcessInfo.processInfo.environment["H3_MODELS_DIR"],
       !fromEnvironment.isEmpty {
        return fromEnvironment
    }
    return FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Caches/minimax-h3-mlx").path
}
