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

        // A practical indexer cannot afford full q rows: it pools each query BLOCK into one
        // vector. Build the block-mean q for each sampled row's own block.
        var blockRowIndices = [Int32]()
        blockRowIndices.reserveCapacity(sampleRows * blockSize)
        for sample in 0..<sampleRows {
            let start = (sample * seq / sampleRows) / blockSize * blockSize
            for offset in 0..<blockSize {
                blockRowIndices.append(Int32(min(start + offset, seq - 1)))
            }
        }
        let qBlockMean = q[0].take(MLXArray(blockRowIndices), axis: 0)
            .reshaped(sampleRows, blockSize, heads, -1)
            .mean(axis: 1)  // (S, heads, D)

        // The vote indexer needs every row of each sampled query block, not just the sampled row.
        let qFullBlocks = q[0].take(MLXArray(blockRowIndices), axis: 0)
            .reshaped(sampleRows, blockSize, heads, -1)  // (S, 64, heads, D)

        // Collect per-head tensors so the head-shared ranking can be evaluated afterwards.
        var massChunks = [MLXArray]()      // (H, S, numBlocks) each
        var proxyChunks = [MLXArray]()     // MSA proxy: block-max of true scores
        var pooledChunks = [MLXArray]()    // practical indexer: block-mean q, block-max of scores
        var votedChunks = [MLXArray]()     // vote indexer: per-row ranking vs pooled keys, block votes

        for headStart in stride(from: 0, to: heads, by: headChunk) {
            let headEnd = min(heads, headStart + headChunk)
            // (H, S, D) x (H, D, seq) -> scores (H, S, seq), fp32.
            let qc = q[0].take(stridedRows, axis: 0)[0..., headStart..<headEnd]
                .transposed(1, 0, 2).asType(.float32)
            let kc = k[0][0..., headStart..<headEnd].transposed(1, 2, 0).asType(.float32)
            func blockView(_ raw: MLXArray, padValue: Float) -> MLXArray {
                var padded = raw
                if padColumns > 0 {
                    let pad = MLXArray.full(
                        [raw.dim(0), raw.dim(1), padColumns], values: MLXArray(padValue))
                    padded = concatenated([padded, pad], axis: -1)
                }
                return padded.reshaped(raw.dim(0), raw.dim(1), numBlocks, blockSize)
            }
            let scores = matmul(qc, kc) * scale
            let probs = softmax(blockView(scores, padValue: -.infinity).reshaped(
                scores.dim(0), scores.dim(1), paddedLength), axis: -1)
            massChunks.append(probs.reshaped(
                scores.dim(0), scores.dim(1), numBlocks, blockSize).sum(axis: -1))
            proxyChunks.append(blockView(scores, padValue: -.infinity).max(axis: -1))

            let qmc = qBlockMean[0..., headStart..<headEnd].transposed(1, 0, 2).asType(.float32)
            let pooledScores = matmul(qmc, kc) * scale
            pooledChunks.append(blockView(pooledScores, padValue: -.infinity).max(axis: -1))

            // Vote indexer: every row of the query block scores the key BLOCKS (keys mean-pooled
            // per block — 1/64th of the full-score cost in production), takes its own top-30 %,
            // and the block's selection is the vote count, ties broken by the mean score.
            let kMean = kc.reshaped(kc.dim(0), kc.dim(1), -1)  // (H, D, seq) -> pool below
            let kPadded = padColumns > 0
                ? concatenated([kMean, MLXArray.zeros([kc.dim(0), kc.dim(1), padColumns])], axis: -1)
                : kMean
            let kBlockMean = kPadded.reshaped(kc.dim(0), kc.dim(1), numBlocks, blockSize)
                .mean(axis: -1)  // (H, D, numBlocks)
            let qRows = qFullBlocks[0..., 0..., headStart..<headEnd]
                .transposed(2, 0, 1, 3).asType(.float32)  // (H, S, 64, D)
            let rowScores = matmul(
                qRows.reshaped(qRows.dim(0), -1, qRows.dim(3)), kBlockMean
            ).reshaped(qRows.dim(0), sampleRows, blockSize, numBlocks)  // (H, S, 64, nB)
            let voteCount = max(1, Int((Double(numBlocks) * 0.3).rounded()))
            let sortedRow = takeAlong(
                rowScores, argSort(rowScores, axis: -1)[.ellipsis, .stride(by: -1)], axis: -1)
            let threshold = sortedRow[.ellipsis, (voteCount - 1)...(voteCount - 1)]
            let votes = (rowScores .>= threshold).asType(.float32).sum(axis: 2)  // (H, S, nB)
            let meanScore = rowScores.mean(axis: 2)
            votedChunks.append(votes * 1000.0 + meanScore)  // rank key: votes first, score ties
            eval(massChunks.last!, proxyChunks.last!, pooledChunks.last!, votedChunks.last!)
        }

        let mass = concatenated(massChunks, axis: 0)      // (heads, S, numBlocks)
        let proxy = concatenated(proxyChunks, axis: 0)
        let pooled = concatenated(pooledChunks, axis: 0)
        let voted = concatenated(votedChunks, axis: 0)

        func meanCurve(orderedBy ranking: MLXArray) -> MLXArray {
            // ranking broadcasts over heads when it is (1, S, numBlocks).
            let order = argSort(ranking, axis: -1)[.ellipsis, .stride(by: -1)]
            let expanded = order.dim(0) == 1
                ? broadcast(order, to: [mass.dim(0), order.dim(1), order.dim(2)]) : order
            let orderedMass = takeAlong(mass, expanded, axis: -1)
            return cumsum(orderedMass.reshaped(-1, numBlocks).mean(axis: 0), axis: 0)
        }
        let oracleCurve = meanCurve(orderedBy: mass)
        let proxyCurve = meanCurve(orderedBy: proxy)
        let pooledCurve = meanCurve(orderedBy: pooled)
        let votedCurve = meanCurve(orderedBy: voted)
        // Head-shared: one selection for all 56 heads (single gather in a real kernel), ranked
        // by the head-mean of the pooled proxy.
        let sharedCurve = meanCurve(orderedBy: pooled.mean(axis: 0).expandedDimensions(axis: 0))
        eval(oracleCurve, proxyCurve, pooledCurve, votedCurve, sharedCurve)

        var report = "[attention audit] layer \(layer) — \(seq) tokens, \(numBlocks) blocks of \(blockSize)\n"
        report += "  blocks kept   oracle    proxy(MSA)  pooled-q   voted     head-shared\n"
        for fraction in [0.05, 0.10, 0.20, 0.30, 0.50] {
            let kept = max(1, Int((Double(numBlocks) * fraction).rounded()))
            report += String(
                format: "  %4d (%3.0f%%)  %8.4f  %8.4f   %8.4f  %8.4f   %8.4f\n",
                kept, fraction * 100,
                oracleCurve[kept - 1].item(Float.self),
                proxyCurve[kept - 1].item(Float.self),
                pooledCurve[kept - 1].item(Float.self),
                votedCurve[kept - 1].item(Float.self),
                sharedCurve[kept - 1].item(Float.self))
        }
        print(report)
    }
}
