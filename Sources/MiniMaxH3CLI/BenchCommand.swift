// BenchCommand.swift - Where a denoising step actually goes, primitive by primitive
// Copyright 2026 Vincent Gourbin
//
// Times the transformer's hot-path primitives on synthetic weights of the real shapes, so the
// cost structure of a step can be mapped in seconds instead of hours — and so an optimization
// can be judged against the primitive it targets rather than against a whole run.
//
// No checkpoint needed: the shapes decide the cost, not the values.

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX
import MLXFast
import MLXNN
import MLXRandom

struct BenchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Time the denoising step's primitives at a given packed-sequence length."
    )

    @Option(name: .long, help: "Packed sequence length (1344x768/124f = 39930, 576x384/124f = 8998).")
    var tokens: Int = 39930

    @Option(name: .long, help: "Weight quantization for the projections: none | qint8 | int4.")
    var quant: String = "qint8"

    @Option(name: .long, help: "Timed repetitions per primitive (after one warmup).")
    var reps: Int = 3

    @Option(name: .long, help: "Measured seconds per step, to express results as a share of it.")
    var stepSeconds: Double = 1081

    @Option(name: .long, help: "Also run the distinct-weights experiment with N weight sets (0 = skip).")
    var weightSets: Int = 0

    func run() async throws {
        let config = H3TransformerConfig()
        let (hidden, heads, headDim) = (config.hiddenSize, config.numAttentionHeads, config.attentionHeadDim)
        let innerDim = heads * headDim
        let blocks = config.numLayers

        print("Packed sequence \(tokens) tokens · hidden \(hidden) · \(heads) heads x \(headDim)"
            + " · ffn \(config.ffnDim) · \(blocks) blocks · weights \(quant)")
        print("Comparing against a measured step of \(String(format: "%.0f", stepSeconds)) s.\n")

        // Affine quantization at the bit width the production loaders use.
        let bits: Int? = quant == "qint8" ? 8 : quant == "int4" ? 4 : nil
        func linear(_ inputDim: Int, _ outputDim: Int) -> any UnaryLayer {
            guard let bits else { return Linear(inputDim, outputDim, bias: false) }
            return QuantizedLinear(inputDim, outputDim, bias: false, groupSize: 64, bits: bits)
        }

        let x = MLXRandom.normal([1, tokens, hidden]).asType(.bfloat16)
        let qkv = (0..<3).map { _ in linear(hidden, innerDim) }
        let outProj = linear(innerDim, hidden)
        let ffProj = linear(hidden, config.ffnDim * 2)
        let ffOut = linear(config.ffnDim, hidden)
        let norm = RMSNorm(dimensions: hidden, eps: config.normEps)
        eval(x)

        var rows = [(name: String, perCall: Double, callsPerStep: Int)]()

        func measure(_ name: String, callsPerStep: Int, _ body: () -> MLXArray) {
            eval(body())  // warmup: compiles kernels, allocates
            let start = Date()
            for _ in 0..<reps { eval(body()) }
            let perCall = Date().timeIntervalSince(start) / Double(reps)
            rows.append((name, perCall, callsPerStep))
            Memory.clearCache()
        }

        // 1. Attention — the quadratic term.
        let q = MLXRandom.normal([1, heads, tokens, headDim]).asType(.bfloat16)
        let k = MLXRandom.normal([1, heads, tokens, headDim]).asType(.bfloat16)
        let v = MLXRandom.normal([1, heads, tokens, headDim]).asType(.bfloat16)
        eval(q, k, v)
        measure("attention (SDPA)", callsPerStep: blocks) {
            MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
        }

        // 1b. The same attention, fed the way production feeds it: transposed views of tensors
        // laid out (B, seq, heads, dim). If the fused kernel wants contiguous inputs, this is
        // where the difference shows up.
        let qSeq = MLXRandom.normal([1, tokens, heads, headDim]).asType(.bfloat16)
        let kSeq = MLXRandom.normal([1, tokens, heads, headDim]).asType(.bfloat16)
        let vSeq = MLXRandom.normal([1, tokens, heads, headDim]).asType(.bfloat16)
        eval(qSeq, kSeq, vSeq)
        measure("attention (production layout)", callsPerStep: blocks) {
            let out = MLXFast.scaledDotProductAttention(
                queries: qSeq.transposed(0, 2, 1, 3),
                keys: kSeq.transposed(0, 2, 1, 3),
                values: vSeq.transposed(0, 2, 1, 3),
                scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
            return out.transposed(0, 2, 1, 3).reshaped(1, tokens, heads * headDim)
        }

        // 2. The block's linear algebra — the linear term.
        measure("q/k/v projections", callsPerStep: blocks) {
            let projected = qkv.map { $0(x) }
            return projected[0] + projected[1] + projected[2]
        }
        measure("attention out projection", callsPerStep: blocks) {
            outProj(MLXRandom.normal([1, tokens, innerDim]).asType(.bfloat16))
        }
        measure("feed-forward", callsPerStep: blocks) {
            let gated = ffProj(x)
            let halves = gated.split(parts: 2, axis: -1)
            return ffOut(halves[0] * sigmoid(halves[1]))
        }

        // 3. The bookkeeping around them — cheap per call, but 52 blocks add up.
        measure("RMSNorm", callsPerStep: 2 * blocks) { norm(x) }

        let rotaryDim = config.rotaryDim
        let cosTable = MLXRandom.normal([tokens, rotaryDim]).asType(.float32)
        let sinTable = MLXRandom.normal([tokens, rotaryDim]).asType(.float32)
        let heads4 = MLXRandom.normal([1, tokens, heads, headDim]).asType(.bfloat16)
        eval(cosTable, sinTable, heads4)
        measure("rotary apply (q and k)", callsPerStep: 2 * blocks) {
            applyH3RotaryEmb(heads4, cos: cosTable, sin: sinTable)
        }

        // AdaLN expands a table of a handful of rows into one value per token, six times.
        let table = MLXRandom.normal([9, hidden]).asType(.bfloat16)
        let indices = MLXArray((0..<tokens).map { Int32($0 % 9) })
        eval(table, indices)
        measure("AdaLN gather + affine", callsPerStep: 6 * blocks) {
            let scale = table.take(indices, axis: 0)
            return x * (1.0 + scale)
        }

        // 4. The whole block, composed exactly as production composes it. Summing primitives
        // misses allocation pressure and every unfused elementwise pass between them, so this
        // is the number that actually predicts a step.
        let modulationTable = (0..<6).map { _ in MLXRandom.normal([9, hidden]).asType(.bfloat16) }
        eval(modulationTable)
        var blockRows = [(name: String, perCall: Double, callsPerStep: Int)]()
        func measureBlock(_ name: String, _ body: () -> MLXArray) {
            eval(body())
            let start = Date()
            for _ in 0..<reps { eval(body()) }
            blockRows.append((name, Date().timeIntervalSince(start) / Double(reps), blocks))
            Memory.clearCache()
        }
        // The same block, wrapped in MLX's graph compiler: fuses the elementwise glue (AdaLN
        // affines, gates, residuals, SwiGLU pointwise) into fewer kernels without touching the
        // math. If this closes the GPU-occupancy gap, opportunity #2 costs one `compile` call.
        func composedBlock(_ input: MLXArray) -> MLXArray {
            let modulation = modulationTable.map { $0.take(indices, axis: 0) }
            var h = input
            let attnInput = norm(h) * (1.0 + modulation[1]) + modulation[0]
            var q = qkv[0](attnInput).reshaped(1, tokens, heads, headDim)
            var k = qkv[1](attnInput).reshaped(1, tokens, heads, headDim)
            let v = qkv[2](attnInput).reshaped(1, tokens, heads, headDim)
            q = applyH3RotaryEmb(q, cos: cosTable, sin: sinTable)
            k = applyH3RotaryEmb(k, cos: cosTable, sin: sinTable)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3),
                scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
            h = h + modulation[2]
                * outProj(attended.transposed(0, 2, 1, 3).reshaped(1, tokens, heads * headDim))
            let ffInput = norm(h) * (1.0 + modulation[4]) + modulation[3]
            let gated = ffProj(ffInput)
            let halves = gated.split(parts: 2, axis: -1)
            return h + modulation[5] * ffOut(halves[0] * sigmoid(halves[1]))
        }
        let compiledBlock = MLX.compile(composedBlock)
        measureBlock("FULL BLOCK (MLX.compile)") { compiledBlock(x) }

        measureBlock("FULL BLOCK (composed)") {
            let modulation = modulationTable.map { $0.take(indices, axis: 0) }
            var h = x
            let attnInput = norm(h) * (1.0 + modulation[1]) + modulation[0]
            var q = qkv[0](attnInput).reshaped(1, tokens, heads, headDim)
            var k = qkv[1](attnInput).reshaped(1, tokens, heads, headDim)
            let v = qkv[2](attnInput).reshaped(1, tokens, heads, headDim)
            q = applyH3RotaryEmb(q, cos: cosTable, sin: sinTable)
            k = applyH3RotaryEmb(k, cos: cosTable, sin: sinTable)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3),
                scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
            h = h + modulation[2]
                * outProj(attended.transposed(0, 2, 1, 3).reshaped(1, tokens, heads * headDim))
            let ffInput = norm(h) * (1.0 + modulation[4]) + modulation[3]
            let gated = ffProj(ffInput)
            let halves = gated.split(parts: 2, axis: -1)
            return h + modulation[5] * ffOut(halves[0] * sigmoid(halves[1]))
        }

        // 5. The decisive experiment for the cost that lives outside the blocks: the same
        // arithmetic, run over N DISTINCT weight sets versus one set reused N times. Identical
        // FLOPs either way — only the weight traffic differs. If distinct is markedly slower,
        // a step is partly paying to stream the transformer's weights, and narrower
        // quantization attacks that directly.
        if weightSets > 0 {
            print("\nDistinct-weights experiment (\(weightSets) sets, identical arithmetic):")
            struct WeightSet {
                let qkv: [any UnaryLayer]
                let out: any UnaryLayer
                let ffProj: any UnaryLayer
                let ffOut: any UnaryLayer
            }
            let sets = (0..<weightSets).map { _ in
                WeightSet(
                    qkv: (0..<3).map { _ in linear(hidden, innerDim) },
                    out: linear(innerDim, hidden),
                    ffProj: linear(hidden, config.ffnDim * 2),
                    ffOut: linear(config.ffnDim, hidden))
            }
            eval(sets.flatMap { [$0.out, $0.ffProj, $0.ffOut] + $0.qkv }.map { ($0 as! Module).parameters() })

            func oneBlock(_ set: WeightSet) -> MLXArray {
                var q = set.qkv[0](x).reshaped(1, tokens, heads, headDim)
                var k = set.qkv[1](x).reshaped(1, tokens, heads, headDim)
                let v = set.qkv[2](x).reshaped(1, tokens, heads, headDim)
                q = applyH3RotaryEmb(q, cos: cosTable, sin: sinTable)
                k = applyH3RotaryEmb(k, cos: cosTable, sin: sinTable)
                let attended = MLXFast.scaledDotProductAttention(
                    queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                    values: v.transposed(0, 2, 1, 3),
                    scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
                let projected = set.out(
                    attended.transposed(0, 2, 1, 3).reshaped(1, tokens, heads * headDim))
                let gated = set.ffProj(x + projected)
                let halves = gated.split(parts: 2, axis: -1)
                return set.ffOut(halves[0] * sigmoid(halves[1]))
            }

            func time(_ label: String, _ body: () -> Void) -> Double {
                body()  // warmup
                let start = Date()
                body()
                let elapsed = Date().timeIntervalSince(start)
                print(String(format: "  %-34@ %8.2f s for %d blocks (%.2f s/block)",
                             label as NSString, elapsed, weightSets, elapsed / Double(weightSets)))
                Memory.clearCache()
                return elapsed
            }

            let reused = time("one weight set, reused") {
                for _ in 0..<weightSets { eval(oneBlock(sets[0])) }
            }
            let distinct = time("distinct weight sets") {
                for set in sets { eval(oneBlock(set)) }
            }
            let perBlockPenalty = (distinct - reused) / Double(weightSets)
            print(String(
                format: "  => streaming penalty %.1f%%, i.e. %.2f s/block -> %.0f s per %d-block step",
                100 * (distinct - reused) / reused, perBlockPenalty,
                perBlockPenalty * Double(blocks), blocks))
        }

        // MARK: - Report

        let total = rows.reduce(0.0) { $0 + $1.perCall * Double($1.callsPerStep) }
        print(String(format: "%-26s %10s %8s %12s %8s", ("primitive" as NSString).utf8String!,
                     ("ms/call" as NSString).utf8String!, ("calls" as NSString).utf8String!,
                     ("s/step" as NSString).utf8String!, ("share" as NSString).utf8String!))
        for row in rows.sorted(by: { $0.perCall * Double($0.callsPerStep) > $1.perCall * Double($1.callsPerStep) }) {
            let perStep = row.perCall * Double(row.callsPerStep)
            print(String(
                format: "%-26@ %10.2f %8d %12.1f %7.1f%%",
                row.name as NSString, row.perCall * 1000, row.callsPerStep, perStep,
                100 * perStep / total))
        }
        print(String(format: "\n%-26@ %31.1f s  (sum of primitives, best case)",
                     "primitives total" as NSString, total))
        for block in blockRows {
            let perStep = block.perCall * Double(block.callsPerStep)
            print(String(format: "%-26@ %31.1f s  (%.2f s/block x %d — %.0f%% of the measured step)",
                         block.name as NSString, perStep, block.perCall, block.callsPerStep,
                         100 * perStep / stepSeconds))
        }
        if blockRows.count == 2 {
            let a = blockRows[0].perCall, b = blockRows[1].perCall
            print(String(format: "%-26@ %30.1f%%  (compile vs eager)",
                         "fusion gain" as NSString, 100 * (b - a) / b))
        }
    }
}
