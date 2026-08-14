// BenchDecodeCommand.swift - Why the video VAE decode leaves the GPU half idle
// Copyright 2026 Vincent Gourbin
//
// The decode phase is structurally simple: whatever the canvas or the duration, it is always N
// INDEPENDENT ViT passes of exactly 1797 tokens (7 latent frames x a 16x16 latent tile + 5
// register/cls tokens), run one after the other with an eval() barrier between them.
//   576x384/124f -> 42 passes    576x384/345f -> 120    1344x768/124f -> 196
// Same shape every time, no data dependency between passes — so batching them is exact, not an
// approximation. This bench answers, on synthetic weights of the real shapes (no checkpoint):
//   1. what arithmetic rate one pass sustains, against the rate a big GEMM sustains here;
//   2. what batching those passes buys (the GEMM grows, the launches and barriers shrink);
//   3. what the per-pass eval() barrier costs on its own;
//   4. what rebuilding the rotary table per pass costs (it is identical for every pass).
//
// Cost of the sweep: ~2 s per tile-pass, so `--batches 1,2,3,6` is about a minute of GPU.

import ArgumentParser
import Foundation
import MiniMaxH3
import MLX
import MLXFast
import MLXNN
import MLXRandom

struct BenchDecodeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-decode",
        abstract: "Time the video VAE's ViT decoder passes and what batching them would buy."
    )

    @Option(name: .long, help: "Batch sizes to sweep (decoder passes fused into one call).")
    var batches: String = "1,2,3,6"

    @Option(name: .long, help: "Decoder passes in the run being modeled (576x384/124f = 42).")
    var tiles: Int = 42

    @Option(name: .long, help: "Timed repetitions per batch size (after one warmup).")
    var reps: Int = 3

    @Option(name: .long, help: "Measured seconds of the decode phase being modeled, for reference.")
    var phaseSeconds: Double = 87.2

    @Flag(name: .long, help: "Also time the blocks under MLX.compile.")
    var compileBlocks = false

    @Flag(name: .long, help: "Time the REAL checkpoint's decode instead of the synthetic sweep.")
    var realVae = false

    @Option(name: .long, help: "Model directory (diffusers layout), for --real-vae.")
    var modelsDir: String = defaultModelsDirectory()

    func run() async throws {
        if realVae {
            try runRealVAE()
            return
        }
        let config = H3VideoVAEConfig()
        let dim = config.decoderNumAttentionHeads * config.decoderAttentionHeadDim
        let heads = config.decoderNumAttentionHeads
        let headDim = config.decoderAttentionHeadDim
        let rotaryDim = Int(Float(headDim) * config.decoderRopeDimRatio)
        let layers = config.decoderNumLayers
        // One tile pass: 7 latent frames x 16x16 latent tile, + register tokens + cls.
        let latentFrames = 7
        let latentSide = 16
        let tokens = latentFrames * latentSide * latentSide + config.decoderNumRegisterTokens + 1

        // Matmul FLOPs of one pass: 2 * tokens * params + attention.
        let paramsPerLayer = 3 * (dim * heads * headDim + heads * headDim) + heads * headDim * dim
            + dim + dim * dim * config.decoderFfnMult * 2 + dim * config.decoderFfnMult * 2
            + dim * config.decoderFfnMult * dim + dim
        let params = layers * paramsPerLayer
        let flopsPerPass = 2.0 * Double(tokens) * Double(params)
            + 4.0 * Double(tokens) * Double(tokens) * Double(dim) * Double(layers)

        print("ViT decoder · \(layers) blocks · dim \(dim) (\(heads)x\(headDim)) · rotary \(rotaryDim)")
        print("One pass: \(tokens) tokens · \(String(format: "%.2f", flopsPerPass / 1e12)) TFLOP · "
            + "weights \(String(format: "%.2f", Double(params) * 2 / 1e9)) GB fp16")
        print("Modeling a decode of \(tiles) passes measured at \(String(format: "%.1f", phaseSeconds)) s "
            + "(\(String(format: "%.2f", phaseSeconds / Double(tiles))) s/pass, "
            + "\(String(format: "%.2f", Double(tiles) * flopsPerPass / 1e12 / phaseSeconds)) TFLOP/s).\n")

        // MARK: Weights (random, real shapes, fp16 like the released decoder)

        func linear(_ inputDim: Int, _ outputDim: Int) -> Linear {
            let layer = Linear(inputDim, outputDim, bias: true)
            layer.update(parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([outputDim, inputDim]).asType(.float16) * 0.02,
                "bias": MLXArray.zeros([outputDim], dtype: .float16),
            ]))
            return layer
        }
        struct Block {
            let norm1: RMSNorm, norm2: RMSNorm
            let toQ: Linear, toK: Linear, toV: Linear, toOut: Linear
            let ffProj: Linear, ffOut: Linear
            let scale1: MLXArray, scale2: MLXArray
        }
        let blocks = (0..<layers).map { _ in
            Block(
                norm1: RMSNorm(dimensions: dim, eps: config.decoderNormEps),
                norm2: RMSNorm(dimensions: dim, eps: config.decoderNormEps),
                toQ: linear(dim, heads * headDim), toK: linear(dim, heads * headDim),
                toV: linear(dim, heads * headDim), toOut: linear(heads * headDim, dim),
                ffProj: linear(dim, dim * config.decoderFfnMult * 2),
                ffOut: linear(dim * config.decoderFfnMult, dim),
                scale1: MLXArray.full([dim], values: MLXArray(Float(0.1))).asType(.float16),
                scale2: MLXArray.full([dim], values: MLXArray(Float(0.1))).asType(.float16))
        }
        eval(blocks.flatMap { [$0.toQ.weight, $0.toK.weight, $0.toV.weight, $0.toOut.weight,
                               $0.ffProj.weight, $0.ffOut.weight, $0.scale1, $0.scale2] })

        // MARK: The rotary table — identical for every pass of every decode

        func buildRotaryTable() -> (MLXArray, MLXArray) {
            let freqCount = rotaryDim / 6
            let invFreq = (0..<freqCount).map {
                1.0 / powf(config.decoderRopeTheta, Float($0) * 6.0 / Float(rotaryDim))
            }
            var angles = [Float](repeating: 0, count: tokens * rotaryDim)
            let axisGrid: (Int) -> [Float] = { size in
                (0..<size).map { 2.0 * ((Float($0) + 0.5) / Float(size)) - 1.0 }
            }
            let (tGrid, hGrid, wGrid) = (axisGrid(latentFrames), axisGrid(latentSide), axisGrid(latentSide))
            let twoPi = 2.0 * Float.pi
            var token = 0
            for t in 0..<latentFrames {
                for h in 0..<latentSide {
                    for w in 0..<latentSide {
                        let base = token * rotaryDim
                        for f in 0..<freqCount {
                            angles[base + f] = twoPi * tGrid[t] * invFreq[f]
                            angles[base + freqCount + f] = twoPi * hGrid[h] * invFreq[f]
                            angles[base + 2 * freqCount + f] = twoPi * wGrid[w] * invFreq[f]
                        }
                        for f in 0..<(3 * freqCount) {
                            angles[base + 3 * freqCount + f] = angles[base + f]
                        }
                        token += 1
                    }
                }
            }
            let table = MLXArray(angles, [tokens, rotaryDim])
            return (cos(table), sin(table))
        }

        let rotaryStart = Date()
        for _ in 0..<10 { _ = buildRotaryTable() }
        let rotaryCost = Date().timeIntervalSince(rotaryStart) / 10
        let (cosTable, sinTable) = buildRotaryTable()
        eval(cosTable, sinTable)

        // MARK: One decoder pass, mirroring H3VideoViTDecoder's block

        func fp32RMSNorm(_ x: MLXArray) -> MLXArray {
            let x32 = x.asType(.float32)
            let scale = rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + config.decoderNormEps)
            return (x32 * scale).asType(x.dtype)
        }
        func runBlock(_ block: Block, _ input: MLXArray) -> MLXArray {
            var x = input
            let batch = x.dim(0)
            var normed = block.norm1(x.asType(.float32)).asType(x.dtype)
            var q = fp32RMSNorm(block.toQ(normed).reshaped(batch, tokens, heads, headDim))
            var k = fp32RMSNorm(block.toK(normed).reshaped(batch, tokens, heads, headDim))
            let v = block.toV(normed).reshaped(batch, tokens, heads, headDim)
            q = applyH3RotaryEmb(q, cos: cosTable, sin: sinTable)
            k = applyH3RotaryEmb(k, cos: cosTable, sin: sinTable)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3),
                scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
            x = x + block.toOut(attended.transposed(0, 2, 1, 3)
                .reshaped(batch, tokens, heads * headDim)) * block.scale1
            normed = block.norm2(x.asType(.float32)).asType(x.dtype)
            let split = block.ffProj(normed).split(parts: 2, axis: -1)
            return x + block.ffOut(split[0] * silu(split[1])) * block.scale2
        }
        let compiled = compileBlocks
            ? blocks.map { block in MLX.compile { (x: MLXArray) in runBlock(block, x) } } : nil
        func decodePass(_ input: MLXArray) -> MLXArray {
            var x = input
            for index in 0..<layers {
                x = compiled.map { $0[index](x) } ?? runBlock(blocks[index], x)
            }
            return x
        }

        // MARK: Sweep

        let sizes = batches.split(separator: ",").compactMap { Int($0) }.filter { $0 >= 1 }
        var baseline: Double?
        print("  batch   s/call    s/pass   TFLOP/s   \(tiles) passes   vs batch 1")
        print("  ─────────────────────────────────────────────────────────────────")
        for size in sizes {
            let input = MLXRandom.normal([size, tokens, dim]).asType(.float16)
            eval(input)
            eval(decodePass(input))  // warmup
            let start = Date()
            for _ in 0..<reps { eval(decodePass(input)) }
            let perCall = Date().timeIntervalSince(start) / Double(reps)
            let perPass = perCall / Double(size)
            let total = perPass * Double(tiles)
            if baseline == nil { baseline = perPass }
            print(String(
                format: "  %5d  %7.3f  %8.3f  %8.2f  %10.1f s   %8.2fx",
                size, perCall, perPass, flopsPerPass / perPass / 1e12, total,
                baseline! / perPass))
            Memory.clearCache()
        }

        // MARK: What the barrier and the rotary rebuild cost on their own

        let input = MLXRandom.normal([1, tokens, dim]).asType(.float16)
        eval(input)
        func timeLoop(evalEach: Bool) -> Double {
            let start = Date()
            var outputs = [MLXArray]()
            for _ in 0..<6 {
                let out = decodePass(input)
                if evalEach { eval(out) } else { outputs.append(out) }
            }
            if !evalEach { eval(outputs) }
            return Date().timeIntervalSince(start) / 6
        }
        _ = timeLoop(evalEach: true)
        let withBarrier = timeLoop(evalEach: true)
        let withoutBarrier = timeLoop(evalEach: false)
        print(String(
            format: "\n  6 passes, eval each: %.3f s/pass   ·   one eval at the end: %.3f s/pass  (%+.1f %%)",
            withBarrier, withoutBarrier, (withoutBarrier / withBarrier - 1) * 100))
        print(String(
            format: "  rotary table rebuild: %.1f ms/pass on the CPU — %.1f s over %d passes, "
                + "identical every time",
            rotaryCost * 1000, rotaryCost * Double(tiles), tiles))

        // MARK: Everything that is NOT the 36 blocks
        //
        // The blocks alone do not account for the measured phase, so time the rest at the real
        // shapes: the patch projection and its 8-D pixel-shuffle permute, the tile stitching, the
        // temporal chunk assembly, and the fp32 post-processing the pipeline runs inside the same
        // phase timer. Shapes below are the 576x384/124f reference run.
        let chunks = tiles / 6
        let pixelFrames = latentFrames * config.temporalCompressionRatio          // 28 per pass
        let tilePx = latentSide * config.spatialCompressionRatio                  // 256
        let (canvasH, canvasW) = (384, 576)
        let outFrames = 124

        var stages = [(name: String, seconds: Double, calls: Int)]()
        func stage(_ name: String, calls: Int, _ body: () -> MLXArray) {
            eval(body())
            let start = Date()
            for _ in 0..<reps { eval(body()) }
            stages.append((name, Date().timeIntervalSince(start) / Double(reps), calls))
            Memory.clearCache()
        }

        let latentTile = MLXRandom.normal([1, config.latentChannels, latentFrames, latentSide, latentSide])
            .asType(.float16)
        let projIn = linear(config.latentChannels, dim)
        let normOut = LayerNorm(dimensions: dim, eps: config.decoderNormEps)
        let projOut = linear(dim, config.outChannels * config.temporalCompressionRatio
            * config.spatialCompressionRatio * config.spatialCompressionRatio)
        let blockOut = MLXRandom.normal([1, tokens, dim]).asType(.float16)
        eval(latentTile, blockOut, projIn.weight, projOut.weight)

        stage("proj_in + register tokens", calls: tiles) {
            let x = projIn(latentTile.transposed(0, 2, 3, 4, 1)
                .reshaped(1, latentFrames * latentSide * latentSide, config.latentChannels))
            return concatenated([x, MLXArray.zeros([1, 5, dim], dtype: .float16)], axis: 1)
        }
        stage("norm_out + proj_out", calls: tiles) {
            projOut(normOut(blockOut.asType(.float32)).asType(.float16))
        }
        let patchTokens = projOut(normOut(blockOut.asType(.float32)).asType(.float16))[
            0..., ..<(latentFrames * latentSide * latentSide)]
        eval(patchTokens)
        stage("pixel-shuffle permute (8-D)", calls: tiles) {
            let ps = config.spatialCompressionRatio
            let pt = config.temporalCompressionRatio
            return patchTokens
                .reshaped(1, latentFrames, latentSide, latentSide, config.outChannels, pt, ps, ps)
                .transposed(0, 4, 1, 5, 2, 6, 3, 7)
                .reshaped(1, config.outChannels, pixelFrames, tilePx, tilePx)
        }

        let decodedTile = MLXRandom.normal([1, config.outChannels, pixelFrames, tilePx, tilePx])
            .asType(.float16)
        eval(decodedTile)
        func blend(_ a: MLXArray, _ b: MLXArray, extent: Int, axis: Int) -> MLXArray {
            let axis = axis < 0 ? a.ndim + axis : axis
            var shape = [Int](repeating: 1, count: a.ndim)
            shape[axis] = extent
            let positions = MLXArray(0..<extent).asType(b.dtype).reshaped(shape) / Float(extent)
            let aTail = a.take(MLXArray((a.dim(axis) - extent)..<a.dim(axis)), axis: axis)
            let bHead = b.take(MLXArray(0..<extent), axis: axis)
            let blended = aTail * (1.0 - positions) + bHead * positions
            let bRest = b.take(MLXArray(extent..<b.dim(axis)), axis: axis)
            return concatenated([blended, bRest], axis: axis)
        }
        stage("tile stitch (2x3 blends + concat)", calls: chunks) {
            var rows = [MLXArray]()
            for row in 0..<2 {
                var tilesInRow = [MLXArray]()
                for column in 0..<3 {
                    var tile = decodedTile
                    if row > 0 { tile = blend(decodedTile, tile, extent: 128, axis: -2) }
                    if column > 0 { tile = blend(decodedTile, tile, extent: 96, axis: -1) }
                    if row < 1 { tile = tile[.ellipsis, ..<(tile.dim(-2) - 128), 0...] }
                    if column < 2 { tile = tile[.ellipsis, ..<(tile.dim(-1) - 96)] }
                    tilesInRow.append(tile)
                }
                rows.append(concatenated(tilesInRow, axis: -1))
            }
            return concatenated(rows, axis: -2)
        }

        let clipPixels = MLXRandom.normal([1, config.outChannels, pixelFrames, canvasH, canvasW])
            .asType(.float16)
        eval(clipPixels)
        stage("temporal blend + chunk concat", calls: 1) {
            var assembled = [MLXArray]()
            for index in 0..<chunks {
                var chunk = clipPixels[0..., 0..., 3..., 0..., 0...]
                if index > 0 { chunk = blend(clipPixels, chunk, extent: 5, axis: 2) }
                assembled.append(chunk)
            }
            return concatenated(assembled, axis: 2)
        }

        let decoded = MLXRandom.normal([1, config.outChannels, outFrames, canvasH, canvasW])
            .asType(.float16)
        eval(decoded)
        stage("post-processing (fp32 denorm, permute, uint8)", calls: 1) {
            let mean = MLXArray(H3Constants.pixelMean).reshaped(1, -1, 1, 1, 1)
            let std = MLXArray(H3Constants.pixelStd).reshaped(1, -1, 1, 1, 1)
            let video = clip(decoded.asType(.float32) * std + mean, min: 0.0, max: 1.0)
            return (video[0].transposed(1, 2, 3, 0) * 255.0).round().asType(.uint8)
        }

        print("\n  Everything around the 36 blocks, at the 576x384/124f shapes:")
        print("  ──────────────────────────────────────────────────────────────")
        var outside = 0.0
        for entry in stages {
            let total = entry.seconds * Double(entry.calls)
            outside += total
            let label = entry.name.padding(toLength: 42, withPad: " ", startingAt: 0)
            print(String(format: "  %@ %7.3f s x%4d = %6.2f s",
                         label, entry.seconds, entry.calls, total))
        }
        let blocksTotal = (baseline ?? 0) * Double(tiles)
        print(String(format: "  %@ %22.2f s",
                     "36 blocks".padding(toLength: 42, withPad: " ", startingAt: 0), blocksTotal))
        print(String(format: "  %@ %22.2f s  (measured phase: %.1f s)",
                     "TOTAL".padding(toLength: 42, withPad: " ", startingAt: 0),
                     blocksTotal + outside, phaseSeconds))
    }

    /// The synthetic sweep models the 36 blocks and everything around them; if it does not add up
    /// to the phase the profiler measured, the difference is in the checkpoint path itself. Two
    /// decodes back to back separate first-touch cost (lazy safetensors paging off the SSD, Metal
    /// kernel specialization) from the steady state that a longer run would actually pay.
    private func runRealVAE() throws {
        print("Loading the video VAE from \(modelsDir) …")
        let loadStart = Date()
        let vae = try H3WeightLoader.loadVideoVAE(modelDirectory: URL(fileURLWithPath: modelsDir))
        eval(vae.parameters())
        print(String(format: "  loaded and resident in %.1f s", Date().timeIntervalSince(loadStart)))

        // Reference run geometry: 576x384 / 124 frames -> 37 latent frames, 24x36 latent canvas.
        let latents = MLXRandom.normal([1, 24, 37, 24, 36]).asType(.float32)
        eval(latents)
        for pass in 1...max(2, reps) {
            let start = Date()
            let decoded = vae.decode(latents)
            eval(decoded)
            let seconds = Date().timeIntervalSince(start)
            print(String(format: "  decode %d: %.1f s  (%.2f s per tile pass, %.2f TFLOP/s)",
                         pass, seconds, seconds / Double(tiles),
                         Double(tiles) * 9.64e12 / 1e12 / seconds))
            Memory.clearCache()
        }
    }
}
