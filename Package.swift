// swift-tools-version: 6.0
// Package.swift - MiniMax-H3 video+audio generation for Apple Silicon (MLX)
// Copyright 2026 Vincent Gourbin

import PackageDescription

let package = Package(
    name: "h3-swift-mlx",
    platforms: [.macOS(.v15)],
    products: [
        // MARK: - Libraries
        .library(name: "MiniMaxH3", targets: ["MiniMaxH3"]),
        // MARK: - CLI Tools
        .executable(name: "minimax-h3", targets: ["MiniMaxH3CLI"]),
    ],
    dependencies: [
        // Pinned exact: mlx-swift has broken API in patch releases before; bump deliberately.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0"),
    ],
    targets: [
        // MARK: - Libraries
        .target(
            name: "MiniMaxH3",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        // MARK: - CLI Tools
        .executableTarget(
            name: "MiniMaxH3CLI",
            dependencies: [
                "MiniMaxH3",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        // MARK: - Tests
        .testTarget(
            name: "MiniMaxH3Tests",
            dependencies: ["MiniMaxH3"]
        ),
    ]
)
