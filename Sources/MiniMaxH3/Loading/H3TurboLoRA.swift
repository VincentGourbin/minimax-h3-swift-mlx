// H3TurboLoRA.swift - Fold a step-distillation LoRA into the transformer weights
// Copyright 2026 Vincent Gourbin
//
// The Turbo LoRAs published for H3 (lightx2v/Minimax-h3-Turbo, and larryvrh's ComfyUI-tree
// equivalent — both Apache-2.0, both used by MiniMaxAI's own demo Space) distil the denoising
// schedule down to 4 or 8 transformer evaluations. Nothing about the scheduler changes: their
// evaluation points are `q_i = (N - i) / N` on the unshifted grid under the same shifts we already
// use (12 video, 3 audio), which is exactly `linspace(1, 0, N+1)` — so NFE 4 is `-s 5` here, and
// NFE 8 is `-s 9`.
//
// The adapter is FOLDED into the weights rather than applied at runtime:
//   W += (alpha / rank) * strength * (B @ A)
// computed in float32 and rounded once on the way back to the checkpoint dtype. Folding keeps the
// forward pass, the quantization path and the prequantized export completely unaware of the LoRA —
// which matters here, because a runtime wrapper would have to survive `H3QuantizationFilter`
// replacing the very Linear layers it wrapped.
//
// ORDER IS THE TRAP: the fold must happen on the bf16 weights BEFORE quantization. Folding onto
// already-quantized weights is not expressible, and a prequantized checkpoint picked up off disk
// has no bf16 stage at all — `loadTransformer` therefore skips that pickup when a LoRA is asked
// for, and pays the slower load instead of silently shipping un-adapted weights.

import Foundation
import MLX

public enum H3TurboLoRA {
    /// Keys look like `transformer_blocks.0.attn.to_q.lora_A.default.weight`; PEFT's `.default.`
    /// is the adapter name and may be absent in other exports.
    private static let suffixA = ".lora_A.weight"
    private static let suffixB = ".lora_B.weight"

    /// Fold every adapter in `url` into `weights`, in place.
    ///
    /// - Parameters:
    ///   - weights: transformer weights, already remapped to this port's module keys.
    ///   - strength: extra multiplier on top of `alpha / rank` (the LoRA cards expose this as a
    ///     sharpness/artifact dial; 1.0 is what their reference inference uses).
    ///   - remap: the same diffusers→ours key rewrite the base weights went through.
    /// - Returns: how many base tensors were modified.
    @discardableResult
    public static func fold(
        into weights: inout [String: MLXArray],
        from url: URL,
        strength: Float = 1.0,
        requireMatch: Bool = true,
        remap: (String) -> String
    ) throws -> Int {
        let (adapter, metadata) = try loadArraysAndMetadata(url: url)

        // Pair A with B. Normalizing `.default.` away lets one code path read both PEFT exports and
        // plain `lora_A/lora_B` ones.
        var pairs = [String: (a: MLXArray, b: MLXArray)]()
        for (rawKey, value) in adapter {
            let key = rawKey.replacingOccurrences(of: ".lora_A.default.", with: ".lora_A.")
                .replacingOccurrences(of: ".lora_B.default.", with: ".lora_B.")
            if key.hasSuffix(Self.suffixA) {
                let base = String(key.dropLast(Self.suffixA.count))
                pairs[base, default: (value, value)].a = value
            } else if key.hasSuffix(Self.suffixB) {
                let base = String(key.dropLast(Self.suffixB.count))
                pairs[base, default: (value, value)].b = value
            }
        }
        guard !pairs.isEmpty else {
            throw H3Error.invalidConfiguration(
                "no lora_A/lora_B tensors in \(url.lastPathComponent) — is this a LoRA?")
        }

        // `alpha` ships in the safetensors metadata and is NOT a constant across releases: the
        // v0.1 file is rank 128 / alpha 8 (scale 0.0625) while v1.0 is rank 128 / alpha 128
        // (scale 1.0). Hardcoding either one silently mis-scales the other.
        let declaredAlpha = metadata["alpha"].flatMap { Float($0) }

        var folded = 0
        var missing = [String]()
        for (base, pair) in pairs.sorted(by: { $0.key < $1.key }) {
            let targetKey = remap(base + ".weight")
            guard let original = weights[targetKey] else {
                missing.append(targetKey)
                continue
            }
            let rank = pair.a.dim(0)
            let scale = strength * (declaredAlpha ?? Float(rank)) / Float(rank)
            let delta = matmul(pair.b.asType(.float32), pair.a.asType(.float32)) * scale
            guard delta.shape == original.shape else {
                throw H3Error.invalidConfiguration(
                    "LoRA \(base): delta \(delta.shape) does not fit weight \(targetKey) "
                        + "\(original.shape)")
            }
            weights[targetKey] = (original.asType(.float32) + delta).asType(original.dtype)
            // Evaluate NOW, one tensor at a time. Deferring to a single eval() at the end builds a
            // lazy graph holding an fp32 copy of every touched weight on top of the 61.7 GB of bf16
            // already resident — the first run of this code was killed by the OS at that point,
            // silently, with no error to read. Per-tensor eval keeps the transient at two copies of
            // the largest weight (~1.2 GB for the feed-forward projection).
            eval(weights[targetKey]!)
            folded += 1
        }

        // A LoRA that matches nothing is a silent no-op otherwise: the run would take the Turbo
        // step count with un-adapted weights and produce noise, which looks like a quality problem
        // rather than a loading problem.
        guard folded > 0 || !requireMatch else {
            throw H3Error.invalidConfiguration(
                "LoRA \(url.lastPathComponent) targets no known weight — first miss: "
                    + (missing.first ?? "n/a"))
        }
        if !missing.isEmpty {
            H3Debug.log("turbo LoRA: \(missing.count) adapter(s) had no target, e.g. \(missing[0])")
        }
        H3Debug.log(
            "turbo LoRA folded into \(folded) tensors from \(url.lastPathComponent)"
                + (declaredAlpha.map { " (alpha \($0))" } ?? ""))
        return folded
    }

    /// The sigma grid point count this port needs for a given number of transformer evaluations.
    /// The Turbo cards quote NFE; `-s` counts grid points, terminal zero included.
    public static func sigmaPoints(forEvaluations evaluations: Int) -> Int { evaluations + 1 }
}
