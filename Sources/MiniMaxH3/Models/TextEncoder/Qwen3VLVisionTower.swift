// Qwen3VLVisionTower.swift - Vision tower of the Qwen3-VL conditioner (fl2va keyframes)
// Copyright 2026 Vincent Gourbin
//
// Port of transformers' Qwen3VLVisionModel: patch embedding (a 2x16x16 full-patch conv3d, i.e.
// a linear from 1536), bilinear-interpolated learned position table (48x48), 2-axis rotary
// (18+18 freqs doubled over the 72-channel heads, applied in fp32), 27 pre-LN blocks, and the
// patch mergers: the final one (pre-shuffle norm, exact GELU) plus three deepstack mergers
// (post-shuffle norm) tapped after blocks 8/16/24 whose outputs are ADDED to the text model's
// hidden states after its first three layers. Token order everywhere is block-major over the
// 2x2 spatial-merge blocks (the image processor flattens patches in that order).

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Qwen3VLVisionConfig: Codable, Sendable {
    public var depth = 27
    public var hiddenSize = 1152
    public var intermediateSize = 4304
    public var numHeads = 16
    public var numPositionEmbeddings = 2304
    public var outHiddenSize = 5120
    public var patchSize = 16
    public var spatialMergeSize = 2
    public var temporalPatchSize = 2
    public var inChannels = 3
    public var deepstackVisualIndexes = [8, 16, 24]

    enum CodingKeys: String, CodingKey {
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case numPositionEmbeddings = "num_position_embeddings"
        case outHiddenSize = "out_hidden_size"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case inChannels = "in_channels"
        case deepstackVisualIndexes = "deepstack_visual_indexes"
    }

    public init() {}

    public var patchDim: Int { inChannels * temporalPatchSize * patchSize * patchSize }  // 1536
    public var headDim: Int { hiddenSize / numHeads }  // 72
    public var gridPerSide: Int { Int(Double(numPositionEmbeddings).squareRoot()) }  // 48

    /// Reads the nested `vision_config` from `text_encoder/config.json`.
    public static func load(from url: URL) throws -> Qwen3VLVisionConfig {
        struct Wrapper: Codable { let vision_config: Qwen3VLVisionConfig }
        return try JSONDecoder().decode(Wrapper.self, from: Data(contentsOf: url)).vision_config
    }
}

// MARK: - Blocks

final class Qwen3VLVisionAttention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(config: Qwen3VLVisionConfig) {
        heads = config.numHeads
        headDim = config.headDim
        _qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
        _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
    }

    /// x (N, hidden); cos/sin (N, headDim) — rotation applied in fp32 (reference contract).
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let tokens = x.dim(0)
        let projected = qkv(x).reshaped(tokens, 3, heads, headDim)
        var q = projected[0..., 0].asType(.float32)
        var k = projected[0..., 1].asType(.float32)
        let v = projected[0..., 2]

        let cosB = cos.reshaped(tokens, 1, headDim)
        let sinB = sin.reshaped(tokens, 1, headDim)
        let half = headDim / 2
        func rotate(_ t: MLXArray) -> MLXArray {
            let x1 = t[.ellipsis, ..<half]
            let x2 = t[.ellipsis, half...]
            return concatenated([-x2, x1], axis: -1)
        }
        q = q * cosB + rotate(q) * sinB
        k = k * cosB + rotate(k) * sinB

        let out = MLXFast.scaledDotProductAttention(
            queries: q.asType(v.dtype).transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.asType(v.dtype).transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: nil
        )
        return proj(out[0].transposed(1, 0, 2).reshaped(tokens, heads * headDim))
    }
}

final class Qwen3VLVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(config: Qwen3VLVisionConfig) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))  // gelu_pytorch_tanh
    }
}

final class Qwen3VLVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: Qwen3VLVisionAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3VLVisionMLP

    init(config: Qwen3VLVisionConfig) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _attn.wrappedValue = Qwen3VLVisionAttention(config: config)
        _mlp.wrappedValue = Qwen3VLVisionMLP(config: config)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var x = x + attn(norm1(x), cos: cos, sin: sin)
        x = x + mlp(norm2(x))
        return x
    }
}

/// Patch merger: groups each 2x2 spatial-merge block (4 consecutive tokens in block-major
/// order) into one 4*hidden vector -> LayerNorm -> fc1 -> EXACT GELU -> fc2 (outHiddenSize).
/// The final merger normalizes pre-shuffle (per token, 1152); deepstack mergers post-shuffle
/// (per merged vector, 4608).
final class Qwen3VLVisionPatchMerger: Module {
    let mergedSize: Int
    let postShuffleNorm: Bool

    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(config: Qwen3VLVisionConfig, postShuffleNorm: Bool) {
        mergedSize = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
        self.postShuffleNorm = postShuffleNorm
        _norm.wrappedValue = LayerNorm(
            dimensions: postShuffleNorm ? mergedSize : config.hiddenSize, eps: 1e-6)
        _fc1.wrappedValue = Linear(mergedSize, mergedSize, bias: true)
        _fc2.wrappedValue = Linear(mergedSize, config.outHiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let merged = postShuffleNorm
            ? norm(x.reshaped(-1, mergedSize))
            : norm(x).reshaped(-1, mergedSize)
        return fc2(gelu(fc1(merged)))  // exact GELU (torch nn.GELU default)
    }
}

// MARK: - Tower

public final class Qwen3VLVisionTower: Module {
    public let config: Qwen3VLVisionConfig

    @ModuleInfo(key: "patch_embed") var patchEmbed: Linear  // full-patch conv3d == linear 1536->1152
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    @ModuleInfo(key: "blocks") var blocks: [Qwen3VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: Qwen3VLVisionPatchMerger
    @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [Qwen3VLVisionPatchMerger]

    public init(config: Qwen3VLVisionConfig) {
        self.config = config
        _patchEmbed.wrappedValue = Linear(config.patchDim, config.hiddenSize, bias: true)
        _posEmbed.wrappedValue = Embedding(
            embeddingCount: config.numPositionEmbeddings, dimensions: config.hiddenSize)
        _blocks.wrappedValue = (0..<config.depth).map { _ in Qwen3VLVisionBlock(config: config) }
        _merger.wrappedValue = Qwen3VLVisionPatchMerger(config: config, postShuffleNorm: false)
        _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
            Qwen3VLVisionPatchMerger(config: config, postShuffleNorm: true)
        }
    }

    /// Block-major token order shared by patches, positions and pos-embed: for merge blocks of
    /// side m, token index runs (hBlock, wBlock, hInBlock, wInBlock).
    static func blockMajorGrid(_ gridH: Int, _ gridW: Int, _ merge: Int) -> [(h: Int, w: Int)] {
        var order = [(Int, Int)]()
        order.reserveCapacity(gridH * gridW)
        for hBlock in 0..<(gridH / merge) {
            for wBlock in 0..<(gridW / merge) {
                for hIn in 0..<merge {
                    for wIn in 0..<merge {
                        order.append((hBlock * merge + hIn, wBlock * merge + wIn))
                    }
                }
            }
        }
        return order
    }

    /// Bilinearly interpolate the 48x48 learned position table onto the (gridH, gridW) token
    /// grid, in block-major order. Small enough to build on CPU.
    func interpolatedPositionEmbeddings(gridH: Int, gridW: Int) -> MLXArray {
        let side = config.gridPerSide
        func axis(_ size: Int) -> [(floor: Int, ceil: Int, frac: Float)] {
            (0..<size).map { index in
                let coordinate = size == 1 ? 0.0 : Double(index) * Double(side - 1) / Double(size - 1)
                let floorValue = Int(coordinate)
                return (floorValue, min(floorValue + 1, side - 1), Float(coordinate - Double(floorValue)))
            }
        }
        let hAxis = axis(gridH)
        let wAxis = axis(gridW)
        let order = Self.blockMajorGrid(gridH, gridW, config.spatialMergeSize)

        var indices = [Int32]()
        var weights = [Float]()
        indices.reserveCapacity(order.count * 4)
        weights.reserveCapacity(order.count * 4)
        for (h, w) in order {
            let (hf, hc, ht) = hAxis[h]
            let (wf, wc, wt) = wAxis[w]
            indices.append(contentsOf: [
                Int32(hf * side + wf), Int32(hf * side + wc),
                Int32(hc * side + wf), Int32(hc * side + wc),
            ])
            weights.append(contentsOf: [
                (1 - ht) * (1 - wt), (1 - ht) * wt, ht * (1 - wt), ht * wt,
            ])
        }
        let table = posEmbed(MLXArray(indices)).reshaped(order.count, 4, config.hiddenSize)
        let weightArray = MLXArray(weights).reshaped(order.count, 4, 1).asType(table.dtype)
        return (table * weightArray).sum(axis: 1)
    }

    /// 2-axis rotary table over the block-major grid: 18 h-frequencies + 18 w-frequencies,
    /// doubled to the 72-channel head. Returns fp32 (cos, sin) of shape (N, 72).
    func rotaryTable(gridH: Int, gridW: Int) -> (cos: MLXArray, sin: MLXArray) {
        let dim = config.headDim / 2  // 36
        let freqCount = dim / 2  // 18
        let invFreq = (0..<freqCount).map { powf(10000.0, -Float(2 * $0) / Float(dim)) }
        let order = Self.blockMajorGrid(gridH, gridW, config.spatialMergeSize)

        var angles = [Float]()
        angles.reserveCapacity(order.count * config.headDim)
        for (h, w) in order {
            var row = [Float](repeating: 0, count: dim)
            for f in 0..<freqCount {
                row[f] = Float(h) * invFreq[f]
                row[freqCount + f] = Float(w) * invFreq[f]
            }
            angles.append(contentsOf: row)
            angles.append(contentsOf: row)  // doubled for rotate_half
        }
        let table = MLXArray(angles, [order.count, config.headDim])
        return (cos(table), sin(table))
    }

    /// patches (N, 1536) in the processor's block-major order -> (imageEmbeds (N/4, 5120),
    /// deepstack features [3 x (N/4, 5120)]).
    public func callAsFunction(
        _ patches: MLXArray, gridH: Int, gridW: Int
    ) -> (embeds: MLXArray, deepstack: [MLXArray]) {
        var x = patchEmbed(patches.asType(patchEmbed.weight.dtype))
        x = x + interpolatedPositionEmbeddings(gridH: gridH, gridW: gridW).asType(x.dtype)
        let (cosTable, sinTable) = rotaryTable(gridH: gridH, gridW: gridW)

        var deepstack = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            x = block(x, cos: cosTable, sin: sinTable)
            if let tap = config.deepstackVisualIndexes.firstIndex(of: index) {
                deepstack.append(deepstackMergers[tap](x))
            }
        }
        return (merger(x), deepstack)
    }
}
