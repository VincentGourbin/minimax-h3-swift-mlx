// H3AttentionAudit.swift - Where does the attention mass actually live?
// Copyright 2026 Vincent Gourbin
//
// Step 0 of the sparse-attention investigation (action plan 405): before implementing anything,
// measure — on a real denoising step with real weights — how concentrated each query row's
// attention mass is over key BLOCKS, and whether MSA's parameter-free selection rule (per-block
// max of the raw scores, then top-k) finds the blocks that carry that mass.
//
// Two curves per audited layer, averaged over sampled query rows and all heads:
//   oracle  — mass captured by the top-k blocks ranked by TRUE mass (upper bound)
//   proxy   — mass captured by the top-k blocks ranked by MSA's block-max score
// If proxy tracks oracle and both saturate early (e.g. 25 % of blocks -> 95 % of mass), a
// block-sparse port is worth building; if mass is flat, plan 405's premise dies here.
//
// Enabled by environment (zero cost otherwise):
//   H3_ATTN_AUDIT=<forward index to audit, 1-based>   e.g. 3 = third denoising step
//   H3_ATTN_AUDIT_LAYERS=0,12,25,37,49                optional, defaults shown
//
// Results are printed to stdout (and the debug log) when the audited forward runs.

import Foundation
import MLX

public final class H3AttentionAudit: @unchecked Sendable {
    public static let shared = H3AttentionAudit()

    public let enabled: Bool
    let targetForward: Int
    let layersToAudit: Set<Int>
    let blockSize = 64
    let sampleRows = 512
    let headChunk = 8

    var forwardIndex = 0
    var currentLayer = -1

    private init() {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["H3_ATTN_AUDIT"], let step = Int(value), step >= 1 {
            enabled = true
            targetForward = step
        } else {
            enabled = false
            targetForward = 0
        }
        if let list = environment["H3_ATTN_AUDIT_LAYERS"] {
            layersToAudit = Set(list.split(separator: ",").compactMap { Int($0) })
        } else {
            layersToAudit = [0, 12, 25, 37, 49]
        }
    }

    /// Called once per transformer forward (i.e. per denoising step).
    func beginForward() {
        forwardIndex += 1
        if forwardIndex == targetForward {
            print("[attention audit] auditing forward \(forwardIndex), layers \(layersToAudit.sorted()), "
                + "block \(blockSize), \(sampleRows) sampled rows")
        }
    }

    /// q/k: (1, seq, heads, headDim), post q/k-norm and post rotary — exactly what SDPA sees.
    func maybeAudit(q: MLXArray, k: MLXArray, scale: Float) {
        guard enabled, forwardIndex == targetForward, layersToAudit.contains(currentLayer) else {
            return
        }
        let layer = currentLayer
        let seq = q.dim(1)
        let heads = q.dim(2)
        let numBlocks = (seq + blockSize - 1) / blockSize
        let stridedRows = MLXArray(
            (0..<sampleRows).map { Int32($0 * seq / sampleRows) })

        // Padded block layout: pad scores with -inf so padded keys carry no mass.
        let paddedLength = numBlocks * blockSize
        let padColumns = paddedLength - seq

        var oracleSum = MLXArray.zeros([numBlocks]).asType(.float32)
        var proxySum = MLXArray.zeros([numBlocks]).asType(.float32)
        var rowsAccumulated = 0

        for headStart in stride(from: 0, to: heads, by: headChunk) {
            let headEnd = min(heads, headStart + headChunk)
            // (H, S, D) x (H, D, seq) -> scores (H, S, seq), fp32.
            let qc = q[0].take(stridedRows, axis: 0)[0..., headStart..<headEnd]
                .transposed(1, 0, 2).asType(.float32)
            let kc = k[0][0..., headStart..<headEnd].transposed(1, 2, 0).asType(.float32)
            var scores = matmul(qc, kc) * scale
            if padColumns > 0 {
                let pad = MLXArray.full(
                    [scores.dim(0), scores.dim(1), padColumns], values: MLXArray(-Float.infinity))
                scores = concatenated([scores, pad], axis: -1)
            }
            let probs = softmax(scores, axis: -1)
            let chunkRows = scores.dim(0) * scores.dim(1)

            // (H, S, numBlocks): true mass per block, and MSA's proxy (block max of raw scores).
            let mass = probs.reshaped(scores.dim(0), scores.dim(1), numBlocks, blockSize)
                .sum(axis: -1)
            let proxy = scores.reshaped(scores.dim(0), scores.dim(1), numBlocks, blockSize)
                .max(axis: -1)

            // Oracle: sort each row's block masses descending, accumulate the mean curve.
            let oracleSorted = takeAlong(mass, argSort(mass, axis: -1)[.ellipsis, .stride(by: -1)], axis: -1)
            // Proxy: order the TRUE mass by the PROXY ranking.
            let proxyOrder = argSort(proxy, axis: -1)[.ellipsis, .stride(by: -1)]
            let massByProxy = takeAlong(mass, proxyOrder, axis: -1)

            oracleSum = oracleSum + oracleSorted.reshaped(-1, numBlocks).sum(axis: 0)
            proxySum = proxySum + massByProxy.reshaped(-1, numBlocks).sum(axis: 0)
            rowsAccumulated += chunkRows
            eval(oracleSum, proxySum)
        }

        let oracleCurve = cumsum(oracleSum / Float(rowsAccumulated), axis: 0)
        let proxyCurve = cumsum(proxySum / Float(rowsAccumulated), axis: 0)
        eval(oracleCurve, proxyCurve)

        var report = "[attention audit] layer \(layer) — \(seq) tokens, \(numBlocks) blocks of \(blockSize)\n"
        report += "  blocks kept   oracle mass   proxy(MSA) mass\n"
        for fraction in [0.05, 0.10, 0.20, 0.30, 0.50] {
            let kept = max(1, Int((Double(numBlocks) * fraction).rounded()))
            let oracle = oracleCurve[kept - 1].item(Float.self)
            let proxy = proxyCurve[kept - 1].item(Float.self)
            report += String(
                format: "  %4d (%3.0f%%)   %8.4f      %8.4f\n",
                kept, fraction * 100, oracle, proxy)
        }
        print(report)
    }
}
