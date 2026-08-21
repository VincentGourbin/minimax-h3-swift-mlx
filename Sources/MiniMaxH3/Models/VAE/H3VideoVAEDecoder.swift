// H3VideoVAEDecoder.swift - Decode path of AutoencoderKLMiniMaxH3 (ViT decoder, f16t4d24)
// Copyright 2026 Vincent Gourbin
//
// Milestone-1 scope: DECODE ONLY. The causal CNN encoder (needed for fl2va keyframes) is deferred,
// which conveniently removes every conv3d — post_quant_conv is 1x1x1 (a per-channel linear) and the
// decoder is a pure ViT. The checkpoint is float32 and the released frames were decoded with
// spatial tiling ON (256px tiles, >=64px overlap, linear blends) and temporal chunking that mirrors
// the encoder's 17-frames -> 5-latents geometry; both are reproduced here.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct H3VideoVAEConfig: Codable, Sendable {
    public var latentChannels = 24
    public var outChannels = 3
    public var blockOutChannels = [128, 256, 256, 512, 512, 1024]
    public var layersPerBlock = 2
    public var normNumGroups = 32
    public var normEps: Float = 1e-6
    public var spatialDownsampleFactors = [2, 2, 2, 2, 1, 1]
    public var temporalDownsampleFactors = [1, 2, 2, 1, 1, 1]
    public var decoderNumLayers = 36
    public var decoderNumAttentionHeads = 32
    public var decoderAttentionHeadDim = 64
    public var decoderNumRegisterTokens = 4
    public var decoderFfnMult = 4
    public var decoderRopeTheta: Float = 100.0
    public var decoderRopeDimRatio: Float = 0.75
    public var decoderNormEps: Float = 1e-5
    public var clipLength = 17
    public var tokenDrop = 3
    public var latentsMean: [Float] = []
    public var latentsStd: [Float] = []

    enum CodingKeys: String, CodingKey {
        case latentChannels = "latent_channels"
        case outChannels = "out_channels"
        case blockOutChannels = "block_out_channels"
        case layersPerBlock = "layers_per_block"
        case normNumGroups = "norm_num_groups"
        case normEps = "norm_eps"
        case spatialDownsampleFactors = "spatial_downsample_factors"
        case temporalDownsampleFactors = "temporal_downsample_factors"
        case decoderNumLayers = "decoder_num_layers"
        case decoderNumAttentionHeads = "decoder_num_attention_heads"
        case decoderAttentionHeadDim = "decoder_attention_head_dim"
        case decoderNumRegisterTokens = "decoder_num_register_tokens"
        case decoderFfnMult = "decoder_ffn_mult"
        case decoderRopeTheta = "decoder_rope_theta"
        case decoderRopeDimRatio = "decoder_rope_dim_ratio"
        case decoderNormEps = "decoder_norm_eps"
        case clipLength = "clip_length"
        case tokenDrop = "token_drop"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
    }

    public init() {}

    public var spatialCompressionRatio: Int { spatialDownsampleFactors.reduce(1, *) }
    public var temporalCompressionRatio: Int { temporalDownsampleFactors.reduce(1, *) }

    public static func load(from url: URL) throws -> H3VideoVAEConfig {
        try JSONDecoder().decode(H3VideoVAEConfig.self, from: Data(contentsOf: url))
    }
}

// MARK: - ViT decoder blocks

/// Affine-free per-head RMSNorm computed in float32 (reference contract).
private func fp32RMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let x32 = x.asType(.float32)
    let scale = rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + eps)
    return (x32 * scale).asType(x.dtype)
}

final class H3VideoDecoderAttention: Module {
    let heads: Int
    let headDim: Int
    let normEps: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(dim: Int, heads: Int, headDim: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        self.normEps = eps
        let innerDim = heads * headDim
        _toQ.wrappedValue = Linear(dim, innerDim, bias: true)
        _toK.wrappedValue = Linear(dim, innerDim, bias: true)
        _toV.wrappedValue = Linear(dim, innerDim, bias: true)
        _toOut.wrappedValue = Linear(innerDim, dim, bias: true)
    }

    /// x (B, N, dim); cos/sin (N, rotaryDim) rotate the leading rotaryDim channels of every head.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, seq) = (x.dim(0), x.dim(1))
        var q = fp32RMSNorm(toQ(x).reshaped(batch, seq, heads, headDim), eps: normEps)
        var k = fp32RMSNorm(toK(x).reshaped(batch, seq, heads, headDim), eps: normEps)
        let v = toV(x).reshaped(batch, seq, heads, headDim)

        q = applyH3RotaryEmb(q, cos: cos, sin: sin)
        k = applyH3RotaryEmb(k, cos: cos, sin: sin)

        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: nil
        )
        return toOut(out.transposed(0, 2, 1, 3).reshaped(batch, seq, heads * headDim))
    }
}

/// diffusers FeedForward(mult=4, activation="swiglu", bias=true): proj -> x * silu(gate) -> out.
final class H3VideoDecoderFeedForward: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(dim: Int, mult: Int) {
        _proj.wrappedValue = Linear(dim, dim * mult * 2, bias: true)
        _out.wrappedValue = Linear(dim * mult, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = proj(x).split(parts: 2, axis: -1)
        return out(split[0] * silu(split[1]))
    }
}

/// Pre-norm block with LayerScale gates (`scale1`/`scale2`); norms run in float32.
final class H3VideoDecoderBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attn: H3VideoDecoderAttention
    @ParameterInfo(key: "scale1") var scale1: MLXArray
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "ff") var ff: H3VideoDecoderFeedForward
    @ParameterInfo(key: "scale2") var scale2: MLXArray

    init(dim: Int, heads: Int, headDim: Int, ffnMult: Int, eps: Float) {
        _norm1.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _attn.wrappedValue = H3VideoDecoderAttention(dim: dim, heads: heads, headDim: headDim, eps: eps)
        _scale1.wrappedValue = MLXArray.zeros([dim])
        _norm2.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _ff.wrappedValue = H3VideoDecoderFeedForward(dim: dim, mult: ffnMult)
        _scale2.wrappedValue = MLXArray.zeros([dim])
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var x = x
        var normed = norm1(x.asType(.float32)).asType(x.dtype)
        x = x + attn(normed, cos: cos, sin: sin) * scale1
        normed = norm2(x.asType(.float32)).asType(x.dtype)
        x = x + ff(normed) * scale2
        return x
    }
}

// MARK: - ViT decoder

/// Non-causal ViT decoder: every latent voxel is one token; 4 register tokens + 1 zero cls token
/// are appended at position 0 and dropped before the patch projection expands each token into a
/// (4, 16, 16) pixel block.
final class H3VideoViTDecoder: Module {
    let config: H3VideoVAEConfig
    let dim: Int
    let rotaryDim: Int

    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ParameterInfo(key: "register_tokens") var registerTokens: MLXArray
    @ModuleInfo(key: "transformer_blocks") var blocks: [H3VideoDecoderBlock]
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(config: H3VideoVAEConfig) {
        self.config = config
        dim = config.decoderNumAttentionHeads * config.decoderAttentionHeadDim
        rotaryDim = Int(Float(config.decoderAttentionHeadDim) * config.decoderRopeDimRatio)

        let dim = self.dim
        _projIn.wrappedValue = Linear(config.latentChannels, dim, bias: true)
        _registerTokens.wrappedValue = MLXArray.zeros([1, config.decoderNumRegisterTokens, dim])
        _blocks.wrappedValue = (0..<config.decoderNumLayers).map { _ in
            H3VideoDecoderBlock(
                dim: dim,
                heads: config.decoderNumAttentionHeads,
                headDim: config.decoderAttentionHeadDim,
                ffnMult: config.decoderFfnMult,
                eps: config.decoderNormEps
            )
        }
        _normOut.wrappedValue = LayerNorm(dimensions: dim, eps: config.decoderNormEps)
        _projOut.wrappedValue = Linear(
            dim,
            config.outChannels * config.temporalCompressionRatio
                * config.spatialCompressionRatio * config.spatialCompressionRatio,
            bias: true
        )
    }

    /// Rotary table over the token grid: per axis, coordinates 2*((i+0.5)/size)-1 in [-1, 1),
    /// angles 2*pi*pos*invFreq with invFreq = 1/theta^arange(0, 1, 2*3/rotaryDim), 3 axes
    /// concatenated then doubled. Register/cls tokens sit at position 0 on every axis.
    private func rotaryTable(frames: Int, height: Int, width: Int) -> (MLXArray, MLXArray) {
        let freqCount = rotaryDim / 6  // rotaryDim = 2 * 3 * freqCount
        let invFreq = (0..<freqCount).map {
            1.0 / powf(config.decoderRopeTheta, Float($0) * 6.0 / Float(rotaryDim))
        }

        let numTokens = frames * height * width
        let suffix = config.decoderNumRegisterTokens + 1
        var angles = [Float](repeating: 0, count: (numTokens + suffix) * rotaryDim)
        let axisGrid: (Int) -> [Float] = { size in
            (0..<size).map { 2.0 * ((Float($0) + 0.5) / Float(size)) - 1.0 }
        }
        let (tGrid, hGrid, wGrid) = (axisGrid(frames), axisGrid(height), axisGrid(width))
        let twoPi = 2.0 * Float.pi
        var token = 0
        for t in 0..<frames {
            for h in 0..<height {
                for w in 0..<width {
                    let base = token * rotaryDim
                    for f in 0..<freqCount {
                        angles[base + f] = twoPi * tGrid[t] * invFreq[f]
                        angles[base + freqCount + f] = twoPi * hGrid[h] * invFreq[f]
                        angles[base + 2 * freqCount + f] = twoPi * wGrid[w] * invFreq[f]
                    }
                    // Duplicate the 3-axis block into the second half (`.tile(2)`).
                    for f in 0..<(3 * freqCount) {
                        angles[base + 3 * freqCount + f] = angles[base + f]
                    }
                    token += 1
                }
            }
        }
        let table = MLXArray(angles, [numTokens + suffix, rotaryDim])
        return (cos(table), sin(table))
    }

    /// z (1, C, F, H, W) latent chunk -> (1, 3, F*4, H*16, W*16) pixels (ImageNet-normalized).
    func callAsFunction(_ z: MLXArray) -> MLXArray {
        let (batch, channels, frames, height, width) = (z.dim(0), z.dim(1), z.dim(2), z.dim(3), z.dim(4))
        var x = z.transposed(0, 2, 3, 4, 1).reshaped(batch, frames * height * width, channels)
        x = projIn(x)
        let numTokens = x.dim(1)

        let cls = MLXArray.zeros([batch, 1, dim], dtype: x.dtype)
        x = concatenated([x, broadcast(registerTokens.asType(x.dtype), to: [batch, config.decoderNumRegisterTokens, dim]), cls], axis: 1)

        let (cosTable, sinTable) = rotaryTable(frames: frames, height: height, width: width)

        for block in blocks {
            x = block(x, cos: cosTable, sin: sinTable)
        }

        x = normOut(x.asType(.float32)).asType(x.dtype)
        x = projOut(x)
        x = x[0..., ..<numTokens]

        let ps = config.spatialCompressionRatio
        let pt = config.temporalCompressionRatio
        x = x.reshaped(batch, frames, height, width, config.outChannels, pt, ps, ps)
        x = x.transposed(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshaped(batch, config.outChannels, frames * pt, height * ps, width * ps)
    }
}

// MARK: - Autoencoder (decode path)

public final class H3VideoVAE: Module {
    public let config: H3VideoVAEConfig

    // Derived temporal-chunking geometry (clip_length 17, ratio 4, token_drop 3):
    // framePrePadding 3, tokensChunkSize 5, tokenOverlap 2, frameOverlap 5.
    let framePrePadding: Int
    let tokensChunkSize: Int
    let tokenOverlap: Int
    let frameOverlap: Int

    // Spatial tiling (released defaults; the released frames are the blended-tile ones).
    public var useTiling = true
    public var tileSampleMinSize = 256
    public var tileSampleMinOverlap = 64

    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Linear?  // 1x1x1 conv3d == channel linear
    @ModuleInfo(key: "decoder") var decoder: H3VideoViTDecoder?
    @ModuleInfo(key: "quant_conv") var quantConv: Linear?
    @ModuleInfo(key: "encoder") var encoder: H3VideoEncoder3d?

    /// - Parameter includeDecoder: the ViT decoder is ~5B parameters. The fl2va keyframe stage
    ///   only encodes, so it builds the VAE without it and skips a ~10 GB read.
    public init(config: H3VideoVAEConfig, includeEncoder: Bool = false, includeDecoder: Bool = true) {
        self.config = config
        let ratio = config.temporalCompressionRatio
        framePrePadding = (ratio - config.clipLength % ratio) % ratio
        tokensChunkSize = (config.clipLength + ratio - 1) / ratio
        tokenOverlap = (tokensChunkSize - config.tokenDrop % tokensChunkSize) % tokensChunkSize
        frameOverlap = max(tokenOverlap * ratio - framePrePadding, 0)

        _postQuantConv.wrappedValue = includeDecoder
            ? Linear(config.latentChannels, config.latentChannels, bias: true) : nil
        _decoder.wrappedValue = includeDecoder ? H3VideoViTDecoder(config: config) : nil
        _quantConv.wrappedValue = includeEncoder
            ? Linear(2 * config.latentChannels, 2 * config.latentChannels, bias: true) : nil
        _encoder.wrappedValue = includeEncoder ? H3VideoEncoder3d(config: config) : nil
    }

    // MARK: - Encode (keyframes)

    /// Encode a visual condition of any length: `AutoencoderKLMiniMaxH3._encode`.
    ///
    /// A single frame has no temporal extent to chunk and goes through the spatial encoder alone —
    /// padding it up to `clipLength` by repetition instead would run the temporal path over 17
    /// copies of the same image and return 2 latent frames rather than 1, which is not the
    /// conditioning MiniMax-H3 was trained with. A frame stack is padded up to a whole number of
    /// 17-frame chunks by repeating its LAST frame, encoded chunk by chunk, and has its 3 trailing
    /// latent frames dropped — which is what turns `17n + 5` pixel frames into `5n + 2` latents.
    ///
    /// - Parameter pixels: `(1, 3, T, H, W)` torch layout, ImageNet-normalized, float32.
    public func encodeVideo(_ pixels: MLXArray) throws -> MLXArray {
        let clipLength = config.clipLength
        let numFrames = pixels.dim(2)
        if numFrames == 1 { return try encodeClip(pixels) }

        var x = pixels
        let padFrames = (clipLength - numFrames % clipLength) % clipLength
        if padFrames > 0 {
            let last = x[0..., 0..., (numFrames - 1)..., 0..., 0...]
            x = concatenated([x] + [MLXArray](repeating: last, count: padFrames), axis: 2)
        }
        var chunks = [MLXArray]()
        for chunk in 0..<(x.dim(2) / clipLength) {
            chunks.append(try encodeClip(x[0..., 0..., (chunk * clipLength)..<((chunk + 1) * clipLength)]))
            eval(chunks[chunks.count - 1])
        }
        var moments = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 2)
        if config.tokenDrop > 0 {
            moments = moments[0..., 0..., ..<(moments.dim(2) - config.tokenDrop)]
        }
        return moments
    }

    /// Encode one temporal clip (a single keyframe for fl2va), spatially tiled like the
    /// reference. pixels: (1, 3, T, H, W) torch layout, ImageNet-normalized, float32.
    /// Returns moments (1, 2*latentChannels, T', hLat, wLat).
    public func encodeClip(_ pixels: MLXArray) throws -> MLXArray {
        guard let encoder, let quantConv else {
            throw H3Error.invalidConfiguration("VAE was loaded without its encoder")
        }
        let ndhwc = pixels.asType(.float32).transposed(0, 2, 3, 4, 1)

        func encodeTile(_ tile: MLXArray) -> MLXArray {
            // encoder output NDHWC -> quant_conv (1x1x1 == channel linear) -> NCDHW
            let moments = quantConv(encoder(tile))
            return moments.transposed(0, 4, 1, 2, 3)
        }

        if !useTiling {
            return encodeTile(ndhwc)
        }
        let height = ndhwc.dim(2)
        let width = ndhwc.dim(3)
        let (yStarts, yLengths, yOverlaps) = splitTiles(
            length: height, tileSize: tileSampleMinSize, minOverlap: tileSampleMinOverlap)
        let (xStarts, xLengths, xOverlaps) = splitTiles(
            length: width, tileSize: tileSampleMinSize, minOverlap: tileSampleMinOverlap)

        var rows = [[MLXArray]]()
        for (yPos, yLen) in zip(yStarts, yLengths) {
            var row = [MLXArray]()
            for (xPos, xLen) in zip(xStarts, xLengths) {
                let tile = ndhwc[
                    0..., 0...,
                    yPos..<min(height, yPos + yLen),
                    xPos..<min(width, xPos + xLen), 0...]
                let encoded = encodeTile(tile)
                eval(encoded)
                row.append(encoded)
            }
            rows.append(row)
        }

        let ratio = config.spatialCompressionRatio
        let latentYOverlaps = yOverlaps.map { $0 / ratio }
        let latentXOverlaps = xOverlaps.map { $0 / ratio }
        var resultRows = [MLXArray]()
        for i in rows.indices {
            var resultRow = [MLXArray]()
            for j in rows[i].indices {
                var tile = rows[i][j]
                if i > 0 { tile = Self.blend(rows[i - 1][j], tile, extent: latentYOverlaps[i - 1], axis: -2) }
                if j > 0 { tile = Self.blend(rows[i][j - 1], tile, extent: latentXOverlaps[j - 1], axis: -1) }
                if i < rows.count - 1 { tile = tile[.ellipsis, ..<(tile.dim(-2) - latentYOverlaps[i]), 0...] }
                if j < rows[i].count - 1 { tile = tile[.ellipsis, ..<(tile.dim(-1) - latentXOverlaps[j])] }
                resultRow.append(tile)
            }
            resultRows.append(concatenated(resultRow, axis: -1))
        }
        return concatenated(resultRows, axis: -2)
    }

    /// Linear cross-fade of `b` over `a`'s trailing `extent` positions along `axis`.
    static func blend(_ a: MLXArray, _ b: MLXArray, extent: Int, axis: Int) -> MLXArray {
        let axis = axis < 0 ? a.ndim + axis : axis  // Swift arrays reject negative indices
        let extent = min(a.dim(axis), b.dim(axis), extent)
        var shape = [Int](repeating: 1, count: a.ndim)
        shape[axis] = extent
        let positions = MLXArray(0..<extent).asType(b.dtype).reshaped(shape) / Float(extent)

        let aTail = a.take(MLXArray((a.dim(axis) - extent)..<a.dim(axis)), axis: axis)
        let bHead = b.take(MLXArray(0..<extent), axis: axis)
        let blended = aTail * (1.0 - positions) + bHead * positions

        if extent == b.dim(axis) { return blended }
        let bRest = b.take(MLXArray(extent..<b.dim(axis)), axis: axis)
        return concatenated([blended, bRest], axis: axis)
    }

    /// Tile layout over `length` pixels: smallest tile count covering the span with every overlap
    /// >= minOverlap; slack distributed round-robin in whole spatial-ratio steps.
    func splitTiles(length: Int, tileSize: Int, minOverlap: Int) -> (starts: [Int], lengths: [Int], overlaps: [Int]) {
        if tileSize >= length { return ([0], [length], []) }
        var numTiles = (length + tileSize - 1) / tileSize
        while tileSize * numTiles - minOverlap * (numTiles - 1) - length < 0 { numTiles += 1 }

        var overlaps = [Int](repeating: minOverlap, count: numTiles - 1)
        let remaining = tileSize * numTiles - overlaps.reduce(0, +) - length
        let ratio = config.spatialCompressionRatio
        for index in 0..<(remaining / ratio) {
            overlaps[index % (numTiles - 1)] += ratio
        }
        var starts = [0]
        for index in 0..<(numTiles - 1) {
            starts.append(starts[starts.count - 1] + tileSize - overlaps[index])
        }
        return (starts, [Int](repeating: tileSize, count: numTiles), overlaps)
    }

    /// Decode one temporal clip, spatially tiled. z (1, C, F, hLat, wLat).
    func decodeClip(_ z: MLXArray) -> MLXArray {
        guard let postQuantConv, let decoder else {
            preconditionFailure("VAE was loaded without its decoder (includeDecoder: false)")
        }
        let projected = postQuantConv(z.transposed(0, 2, 3, 4, 1)).transposed(0, 4, 1, 2, 3)
        if !useTiling { return decoder(projected) }

        let ratio = config.spatialCompressionRatio
        let height = z.dim(-2) * ratio
        let width = z.dim(-1) * ratio
        let (yStarts, yLengths, yOverlaps) = splitTiles(
            length: height, tileSize: tileSampleMinSize, minOverlap: tileSampleMinOverlap)
        let (xStarts, xLengths, xOverlaps) = splitTiles(
            length: width, tileSize: tileSampleMinSize, minOverlap: tileSampleMinOverlap)

        var rows = [[MLXArray]]()
        for (yPos, yLen) in zip(yStarts, yLengths) {
            var row = [MLXArray]()
            for (xPos, xLen) in zip(xStarts, xLengths) {
                let tile = projected[
                    0..., 0..., 0...,
                    (yPos / ratio)..<min(z.dim(-2), yPos / ratio + yLen / ratio),
                    (xPos / ratio)..<min(z.dim(-1), xPos / ratio + xLen / ratio)
                ]
                let decoded = decoder(tile)
                eval(decoded)
                row.append(decoded)
            }
            rows.append(row)
        }

        // Stitch with linear blends, trimming each tile's trailing overlap.
        var resultRows = [MLXArray]()
        for i in rows.indices {
            var resultRow = [MLXArray]()
            for j in rows[i].indices {
                var tile = rows[i][j]
                if i > 0 { tile = Self.blend(rows[i - 1][j], tile, extent: yOverlaps[i - 1], axis: -2) }
                if j > 0 { tile = Self.blend(resultRowSource(rows[i], j - 1), tile, extent: xOverlaps[j - 1], axis: -1) }
                if i < rows.count - 1 { tile = tile[.ellipsis, ..<(tile.dim(-2) - yOverlaps[i]), 0...] }
                if j < rows[i].count - 1 { tile = tile[.ellipsis, ..<(tile.dim(-1) - xOverlaps[j])] }
                resultRow.append(tile)
            }
            resultRows.append(concatenated(resultRow, axis: -1))
        }
        return concatenated(resultRows, axis: -2)
    }

    private func resultRowSource(_ row: [MLXArray], _ index: Int) -> MLXArray { row[index] }

    /// Decode a latent video (1, C, T, hLat, wLat) -> pixels (1, 3, frames, H, W), mirroring the
    /// encoder's chunking: chunks of 5 latent tokens decoded with 2 tokens of lookahead, 3 leading
    /// pixel frames cut per clip, 5-frame linear cross-fade between consecutive chunks.
    public func decode(_ z: MLXArray) -> MLXArray {
        let temporalRatio = config.temporalCompressionRatio
        let chunkNumFrames = tokensChunkSize * temporalRatio

        guard let postQuantConv else {
            preconditionFailure("VAE was loaded without its decoder (includeDecoder: false)")
        }
        // Follow the weights' compute dtype (fp16 per the released autocast recipe) — a fp32
        // input would silently promote every matmul back to fp32.
        var z = z.asType(postQuantConv.weight.dtype)
        let numTokens = z.dim(2) + config.tokenDrop
        let padTokens = (tokensChunkSize - numTokens % tokensChunkSize) % tokensChunkSize
        let numChunks = (numTokens + padTokens) / tokensChunkSize - (config.tokenDrop > 0 ? 1 : 0)
        let tokensBeforePad = z.dim(2)
        if padTokens > 0 {
            let lastToken = z[0..., 0..., (z.dim(2) - 1)..., 0..., 0...]
            let repeats = (0..<padTokens).map { _ in lastToken }
            z = concatenated([z] + repeats, axis: 2)
        }

        var decodedChunks = [MLXArray]()
        var overlap: MLXArray? = nil
        for index in 0..<numChunks {
            let start = index * tokensChunkSize
            let end = min(z.dim(2), start + tokensChunkSize + tokenOverlap)
            let clip = decodeClip(z[0..., 0..., start..<end, 0..., 0...])
            let passes = config.tokenDrop > 0 ? 2 : 1
            for j in 0..<passes {
                let frameStart = j * chunkNumFrames
                let frameEnd = min(clip.dim(2), frameStart + chunkNumFrames)
                var chunk = clip[0..., 0..., frameStart..<frameEnd, 0..., 0...]
                chunk = chunk[0..., 0..., framePrePadding..., 0..., 0...]
                if j == 0 {
                    if let previous = overlap {
                        chunk = Self.blend(previous, chunk, extent: frameOverlap, axis: 2)
                    }
                    eval(chunk)
                    decodedChunks.append(chunk)
                } else {
                    overlap = chunk
                }
            }
        }
        if let previous = overlap { decodedChunks.append(previous) }

        var decoded = concatenated(decodedChunks, axis: 2)

        if padTokens > 0 {
            // Trailing pixel frames from the repeated latents: a chunk's final latent token covers
            // only clip_length % ratio pixel frames, the others cover the full ratio.
            let intraTail = config.clipLength % temporalRatio
            var padFrames = 0
            for k in 0..<padTokens {
                let isChunkTail = intraTail != 0 && (tokensBeforePad + k) % tokensChunkSize == 0
                padFrames += isChunkTail ? intraTail : temporalRatio
            }
            decoded = decoded[0..., 0..., ..<(decoded.dim(2) - padFrames), 0..., 0...]
        }
        return decoded
    }
}
