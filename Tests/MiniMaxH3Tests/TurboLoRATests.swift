// TurboLoRATests.swift - Folding a step-distillation LoRA into the transformer weights
// Copyright 2026 Vincent Gourbin
//
// No checkpoint needed: the fold is arithmetic on tensors of the right shape, and the failure
// modes worth pinning are the ones that would otherwise be invisible at runtime — a mis-scaled
// alpha, a key rewrite that silently matches nothing, a wrong multiplication order.

import Foundation
import MLX
import Testing

@testable import MiniMaxH3

@Suite struct TurboLoRATests {
    /// The same rewrite `loadTransformer` applies to the base weights.
    private func remap(_ key: String) -> String {
        key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj.")
            .replacingOccurrences(of: ".ff.net.2.", with: ".ff.out.")
    }

    /// Write a one-adapter LoRA file in the layout the published checkpoints use.
    private func writeLoRA(
        base: String, a: MLXArray, b: MLXArray, alpha: String?, peftSuffix: Bool = true
    ) throws -> URL {
        let name = peftSuffix ? "\(base).lora_A.default.weight" : "\(base).lora_A.weight"
        let nameB = peftSuffix ? "\(base).lora_B.default.weight" : "\(base).lora_B.weight"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-\(UUID().uuidString).safetensors")
        try save(arrays: [name: a, nameB: b], metadata: alpha.map { ["alpha": $0] } ?? [:], url: url)
        return url
    }

    @Test func foldsDeltaWithAlphaOverRankScale() throws {
        // rank 2, alpha 1 -> scale 0.5, so the fold must halve B @ A.
        let a = MLXArray(converting: [1.0, 0.0, 0.0, 1.0], [2, 2]).asType(.bfloat16)
        let b = MLXArray(converting: [2.0, 0.0, 0.0, 4.0], [2, 2]).asType(.bfloat16)
        let url = try writeLoRA(base: "transformer_blocks.0.attn.to_q", a: a, b: b, alpha: "1")
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.0.attn.to_q.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        let folded = try H3TurboLoRA.fold(into: &weights, from: url, remap: remap)

        #expect(folded == 1)
        let result = weights["transformer_blocks.0.attn.to_q.weight"]!.asType(.float32)
        #expect(result[0, 0].item(Float.self) == 1.0)   // 2 * 0.5
        #expect(result[1, 1].item(Float.self) == 2.0)   // 4 * 0.5
        #expect(result[0, 1].item(Float.self) == 0.0)
        #expect(weights["transformer_blocks.0.attn.to_q.weight"]!.dtype == .bfloat16)
    }

    @Test func absentAlphaMeansScaleOne() throws {
        let a = MLXArray.ones([2, 2]).asType(.bfloat16)
        let b = MLXArray.ones([2, 2]).asType(.bfloat16)
        let url = try writeLoRA(base: "transformer_blocks.0.attn.to_k", a: a, b: b, alpha: nil)
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.0.attn.to_k.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        try H3TurboLoRA.fold(into: &weights, from: url, remap: remap)
        // B @ A with both all-ones at rank 2 = 2 everywhere, scale 1.
        #expect(weights["transformer_blocks.0.attn.to_k.weight"]!.asType(.float32)[0, 0]
            .item(Float.self) == 2.0)
    }

    @Test func strengthMultipliesOnTopOfAlpha() throws {
        let a = MLXArray.ones([1, 2]).asType(.bfloat16)
        let b = MLXArray.ones([2, 1]).asType(.bfloat16)
        let url = try writeLoRA(base: "transformer_blocks.1.ff.net.2", a: a, b: b, alpha: "1")
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.1.ff.out.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        try H3TurboLoRA.fold(into: &weights, from: url, strength: 3.0, remap: remap)
        #expect(weights["transformer_blocks.1.ff.out.weight"]!.asType(.float32)[0, 0]
            .item(Float.self) == 3.0)
    }

    /// `.to_out.0.` and `.ff.net.*` only reach our modules through the loader's rewrite; if that
    /// rewrite is not applied to the adapter keys too, every target misses.
    @Test func adapterKeysGoThroughTheSameRemapAsBaseWeights() throws {
        let a = MLXArray.ones([1, 2]).asType(.bfloat16)
        let b = MLXArray.ones([2, 1]).asType(.bfloat16)
        let url = try writeLoRA(base: "transformer_blocks.0.attn.to_out.0", a: a, b: b, alpha: "1")
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.0.attn.to_out.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        #expect(try H3TurboLoRA.fold(into: &weights, from: url, remap: remap) == 1)
    }

    @Test func plainLoraKeysWithoutPeftAdapterNameAlsoLoad() throws {
        let a = MLXArray.ones([1, 2]).asType(.bfloat16)
        let b = MLXArray.ones([2, 1]).asType(.bfloat16)
        let url = try writeLoRA(
            base: "transformer_blocks.0.attn.to_v", a: a, b: b, alpha: "1", peftSuffix: false)
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.0.attn.to_v.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        #expect(try H3TurboLoRA.fold(into: &weights, from: url, remap: remap) == 1)
    }

    /// An adapter that matches nothing must fail loudly: silently folding zero tensors would run
    /// the Turbo step count against un-adapted weights, which looks like a quality problem.
    @Test func adapterMatchingNothingThrows() throws {
        let a = MLXArray.ones([1, 2]).asType(.bfloat16)
        let b = MLXArray.ones([2, 1]).asType(.bfloat16)
        let url = try writeLoRA(base: "some.other.tree.linear", a: a, b: b, alpha: "1")
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.0.attn.to_q.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        #expect(throws: H3Error.self) {
            try H3TurboLoRA.fold(into: &weights, from: url, remap: remap)
        }
    }

    @Test func shapeMismatchThrowsInsteadOfBroadcasting() throws {
        let a = MLXArray.ones([1, 3]).asType(.bfloat16)
        let b = MLXArray.ones([2, 1]).asType(.bfloat16)
        let url = try writeLoRA(base: "transformer_blocks.0.attn.to_q", a: a, b: b, alpha: "1")
        defer { try? FileManager.default.removeItem(at: url) }

        var weights = ["transformer_blocks.0.attn.to_q.weight": MLXArray.zeros([2, 2], dtype: .bfloat16)]
        #expect(throws: H3Error.self) {
            try H3TurboLoRA.fold(into: &weights, from: url, remap: remap)
        }
    }

    /// NFE is what the LoRA cards quote; `-s` counts grid points including the terminal zero.
    @Test func sigmaPointsMatchTheTurboEvaluationCounts() {
        #expect(H3TurboLoRA.sigmaPoints(forEvaluations: 4) == 5)
        #expect(H3TurboLoRA.sigmaPoints(forEvaluations: 8) == 9)
    }

    /// The distilled grids quoted by the Turbo repo must be what our scheduler already produces:
    /// video [1, 0.9730, 0.9231, 0.8000] and audio [1, 0.9000, 0.7500, 0.5000] at NFE 4.
    @Test func schedulerReproducesThePublishedTurboGrid() throws {
        let video = H3Scheduler(shift: 12.0)
        try video.setTimesteps(numInferenceSteps: H3TurboLoRA.sigmaPoints(forEvaluations: 4))
        let expectedVideo: [Float] = [1.0, 0.9730, 0.9231, 0.8000, 0.0]
        #expect(video.sigmas.count == expectedVideo.count)
        for (got, want) in zip(video.sigmas, expectedVideo) {
            #expect(abs(got - want) < 5e-4)
        }

        let audio = H3Scheduler(shift: 3.0)
        try audio.setTimesteps(numInferenceSteps: H3TurboLoRA.sigmaPoints(forEvaluations: 4))
        let expectedAudio: [Float] = [1.0, 0.9000, 0.7500, 0.5000, 0.0]
        for (got, want) in zip(audio.sigmas, expectedAudio) {
            #expect(abs(got - want) < 5e-4)
        }
    }
}
