// KeyframeConditioningTests.swift - fl2va invariants that need no checkpoint
// Copyright 2026 Vincent Gourbin
//
// These cover the parts of keyframe conditioning that are pure geometry or pure arithmetic:
// the interleaved-mrope layout, the presentation's two taggings, canvas preparation, the
// block-major patchify, and the packed sequence's condition rows. They run in milliseconds,
// so a review can check the traps without the 163 GB checkpoint. Run with
// `xcodebuild test -scheme minimax-h3 -destination 'platform=macOS'` — `swift test` cannot load
// MLX's metallib. The checkpoint-backed smoke tests live in SmokeTests.swift.

import MLX
import Testing
import Tokenizers
@testable import MiniMaxH3

// MARK: - Interleaved mrope layout

@Suite struct MultimodalLayoutTests {
    /// Grid 4x6 patches -> 2x3 merged tokens, i.e. a 6-token image run.
    private static func layout() throws -> Qwen3VLMultimodalLayout {
        // [text, text, <|vision_start|>, 6 image pads, <|vision_end|>, text]
        // The delimiters are TEXT for mrope purposes — only the pads are type 1.
        let types: [Int32] = [0, 0, 0] + [Int32](repeating: 1, count: 6) + [0, 0]
        return try Qwen3VLMultimodalLayout(mmTokenTypes: types, imageGrids: [(h: 4, w: 6)])
    }

    @Test func textRunsAdvanceAllThreeAxesTogether() throws {
        let positions = try Self.layout().positions
        // Leading text run: three tokens at 0, 1, 2 on t, h and w alike.
        for index in 0..<3 {
            for axis in 0..<3 {
                #expect(positions[axis, index].item(Int32.self) == Int32(index))
            }
        }
    }

    @Test func imageRunFreezesTimeAndSpreadsOverTheMergedGrid() throws {
        let positions = try Self.layout().positions
        // The run starts at 3 (after the two labels and vision_start).
        let start = 3
        for token in 0..<6 {
            #expect(positions[0, start + token].item(Int32.self) == 3)  // t frozen
        }
        // h/w run over the 2x3 merged grid in row-major order, offset by the current position.
        let expectedH: [Int32] = [3, 3, 3, 4, 4, 4]
        let expectedW: [Int32] = [3, 4, 5, 3, 4, 5]
        for token in 0..<6 {
            #expect(positions[1, start + token].item(Int32.self) == expectedH[token])
            #expect(positions[2, start + token].item(Int32.self) == expectedW[token])
        }
    }

    /// THE trap: after an image, the clock advances by the larger merged side — not by the
    /// number of image tokens. Six tokens here, but the grid is 2x3, so the next text token
    /// sits at 3 + max(2, 3) = 6, not at 3 + 6 = 9.
    @Test func clockAdvancesByLargestMergedSideNotTokenCount() throws {
        let positions = try Self.layout().positions
        // The image run occupies indices 3...8; index 9 is <|vision_end|>, the first token to
        // resume the text clock. It sits at 3 + max(2, 3) = 6 — a naive "advance by token
        // count" would put it at 3 + 6 = 9.
        let resume = 9
        for axis in 0..<3 {
            #expect(positions[axis, resume].item(Int32.self) == 6)
            #expect(positions[axis, resume + 1].item(Int32.self) == 7)
        }
    }

    @Test func imageRunsAreReported() throws {
        let runs = try Self.layout().imageRuns
        #expect(runs.count == 1)
        #expect(runs[0].start == 3 && runs[0].count == 6)
    }

    @Test func mismatchedGridIsRejected() throws {
        // A 6-token run cannot come from a 4x4 grid (which merges to 4 tokens).
        let types: [Int32] = [0] + [Int32](repeating: 1, count: 6) + [0]
        #expect(throws: H3Error.self) {
            try Qwen3VLMultimodalLayout(mmTokenTypes: types, imageGrids: [(h: 4, w: 4)])
        }
    }

    @Test func unusedGridIsRejected() throws {
        #expect(throws: H3Error.self) {
            try Qwen3VLMultimodalLayout(mmTokenTypes: [0, 0], imageGrids: [(h: 4, w: 4)])
        }
    }
}

// MARK: - Presentation

/// Minimal tokenizer double: one id per word plus the three Qwen vision specials. Enough for
/// the presentation, which only encodes text and looks up special tokens.
private struct StubTokenizer: Tokenizer {
    static let specials = [
        "<|vision_start|>": 1000, "<|image_pad|>": 1001, "<|vision_end|>": 1002,
    ]

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map(String.init)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // Deterministic and collision-free enough for layout assertions: one id per word,
        // never colliding with the special ids above.
        tokenize(text: text).enumerated().map { index, _ in 10 + index }
    }

    func encode(text: String) -> [Int] { encode(text: text, addSpecialTokens: true) }
    func decode(tokens: [Int]) -> String { decode(tokens: tokens, skipSpecialTokens: false) }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { Self.specials[token] }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map(convertTokenToId) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map(convertIdToToken) }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    // H3 never applies a chat template — the prompt is presented verbatim. The protocol
    // requires the whole family, so every overload refuses loudly.
    private func noChatTemplate() throws -> [Int] {
        throw H3Error.invalidInput("the H3 presentation never applies a chat template")
    }

    func applyChatTemplate(messages: [Message]) throws -> [Int] { try noChatTemplate() }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        try noChatTemplate()
    }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { try noChatTemplate() }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try noChatTemplate()
    }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try noChatTemplate()
    }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
    ) throws -> [Int] { try noChatTemplate() }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { try noChatTemplate() }
}

@Suite struct PresentationTests {
    @Test func textOnlyPresentationIsTheVerbatimPrompt() throws {
        let presentation = try H3Presentation(
            prompt: "a b c", tokenizer: StubTokenizer())
        #expect(presentation.tokenIds.count == 3)
        #expect(presentation.tokenTags.allSatisfy { $0 == H3Constants.textTag })
        #expect(presentation.mmTokenTypes.allSatisfy { $0 == 0 })
        #expect(presentation.imageGrids.isEmpty)
        #expect(try presentation.multimodalLayout() == nil)
    }

    /// The two taggings must not be conflated: H3's AdaLN row tags mark the WHOLE vision block
    /// (delimiters included) as video, while the mrope token types mark ONLY the image pads.
    @Test func visionBlockTaggingDiffersBetweenAdaLNAndMrope() throws {
        let presentation = try H3Presentation(
            prompt: "hello world",
            imageGrids: [(h: 4, w: 6)],  // -> 6 merged tokens
            tokenizer: StubTokenizer())

        let labelCount = presentation.tokenIds.count - (6 + 2) - 2  // total - vision block - prompt
        #expect(labelCount > 0)

        let blockRange = labelCount..<(labelCount + 6 + 2)
        for index in blockRange {
            #expect(presentation.tokenTags[index] == H3Constants.videoTag)
        }
        // mrope types: delimiters are text, pads are image.
        #expect(presentation.mmTokenTypes[blockRange.lowerBound] == 0)          // vision_start
        #expect(presentation.mmTokenTypes[blockRange.upperBound - 1] == 0)      // vision_end
        for index in (blockRange.lowerBound + 1)..<(blockRange.upperBound - 1) {
            #expect(presentation.mmTokenTypes[index] == 1)
        }
        // Label and prompt rows stay text on both taggings.
        #expect(presentation.tokenTags[0] == H3Constants.textTag)
        #expect(presentation.tokenTags.last == H3Constants.textTag)
    }

    @Test func visionBlockCarriesOnePadPerMergedToken() throws {
        let presentation = try H3Presentation(
            prompt: "x", imageGrids: [(h: 8, w: 4)], tokenizer: StubTokenizer())
        let pads = presentation.tokenIds.filter { $0 == Int32(StubTokenizer.specials["<|image_pad|>"]!) }
        #expect(pads.count == (8 / 2) * (4 / 2))
        #expect(presentation.tokenIds.contains(Int32(StubTokenizer.specials["<|vision_start|>"]!)))
        #expect(presentation.tokenIds.contains(Int32(StubTokenizer.specials["<|vision_end|>"]!)))
    }

    @Test func presentationFeedsAConsistentMropeLayout() throws {
        let presentation = try H3Presentation(
            prompt: "a b", imageGrids: [(h: 4, w: 6)], tokenizer: StubTokenizer())
        let layout = try #require(try presentation.multimodalLayout())
        #expect(layout.positions.dim(1) == presentation.tokenIds.count)
        #expect(layout.imageRuns.map(\.count) == [6])
    }
}

// MARK: - Canvas preparation and patchify

@Suite struct KeyframeImageTests {
    /// Solid-colour image of the given size.
    private static func solid(width: Int, height: Int, rgb: (UInt8, UInt8, UInt8)) -> H3KeyframeImage {
        var pixels = [UInt8]()
        pixels.reserveCapacity(width * height * 3)
        for _ in 0..<(width * height) { pixels += [rgb.0, rgb.1, rgb.2] }
        return H3KeyframeImage(width: width, height: height, pixels: pixels)
    }

    @Test func canvasSizedImagePassesThroughUntouched() {
        let image = Self.solid(width: 64, height: 32, rgb: (10, 20, 30))
        let prepared = image.prepared(canvasWidth: 64, canvasHeight: 32, stretch: true)
        #expect(prepared.pixels == image.pixels)
    }

    @Test func lanczosPreservesAConstantImage() {
        // A resampling filter with normalized taps must leave a flat field flat — this catches
        // coefficient-normalization and rounding mistakes.
        let resized = Self.solid(width: 100, height: 60, rgb: (7, 128, 255))
            .resizedLanczos(toWidth: 37, toHeight: 23)
        #expect(resized.width == 37 && resized.height == 23)
        #expect(resized.pixels.count == 37 * 23 * 3)
        for pixel in stride(from: 0, to: resized.pixels.count, by: 3) {
            #expect(resized.pixels[pixel] == 7)
            #expect(resized.pixels[pixel + 1] == 128)
            #expect(resized.pixels[pixel + 2] == 255)
        }
    }

    @Test func coverCropKeepsTheCanvasAndCentresIt() {
        // 100x50 onto 40x40: scale = max(0.4, 0.8) = 0.8 -> 80x40 -> centre crop 40 wide.
        let prepared = Self.solid(width: 100, height: 50, rgb: (1, 2, 3))
            .prepared(canvasWidth: 40, canvasHeight: 40, stretch: false)
        #expect(prepared.width == 40 && prepared.height == 40)
    }

    @Test func patchesAreNormalizedAndTemporallyDuplicated() {
        // White image -> (255/255 - 0.5) / 0.5 = +1 everywhere.
        let (patches, gridH, gridW) = Self.solid(width: 32, height: 32, rgb: (255, 255, 255))
            .visionPatches()
        #expect(gridH == 2 && gridW == 2)
        #expect(patches.shape == [4, 1536])
        #expect(abs(patches.max().item(Float.self) - 1.0) < 1e-6)
        #expect(abs(patches.min().item(Float.self) - 1.0) < 1e-6)

        // Per patch the layout is [channel][t][row][col]: the two temporal copies are identical.
        let black = Self.solid(width: 32, height: 32, rgb: (0, 0, 0)).visionPatches().patches
        #expect(abs(black.max().item(Float.self) + 1.0) < 1e-6)
        let firstCopy = patches[0, 0..<256]
        let secondCopy = patches[0, 256..<512]
        #expect(abs((firstCopy - secondCopy).abs().max().item(Float.self)) < 1e-6)
    }

    /// Token order is block-major over the 2x2 merge blocks — the order the reference image
    /// processor emits, and the order the position table and rotary grid assume.
    @Test func patchOrderIsBlockMajor() {
        // 64x32 -> 4x2 patch grid -> one 2x2 merge block per (blockRow, blockCol).
        // Paint each 16x16 patch with a unique red level so rows are identifiable.
        let (width, height, patch) = (64, 32, 16)
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let patchIndex = (y / patch) * (width / patch) + (x / patch)
                pixels[(y * width + x) * 3] = UInt8(patchIndex * 20)
            }
        }
        let image = H3KeyframeImage(width: width, height: height, pixels: pixels)
        let (patches, gridH, gridW) = image.visionPatches()
        #expect(gridH == 2 && gridW == 4)

        let expected = Qwen3VLVisionTower.blockMajorGrid(gridH, gridW, 2)
            .map { $0.h * gridW + $0.w }
        for (row, patchIndex) in expected.enumerated() {
            // Channel 0, temporal copy 0, first pixel of the patch.
            let value = patches[row, 0].item(Float.self)
            let expectedValue = (Float(patchIndex * 20) / 255.0 - 0.5) / 0.5
            #expect(abs(value - expectedValue) < 1e-3)
        }
    }
}

// MARK: - Packed sequence with keyframe conditioning

@Suite struct KeyframePackingTests {
    private static let latentFrames = 7
    private static let (latentHeight, latentWidth) = (8, 6)
    private static let rowsPerFrame = (8 / 2) * (6 / 2)

    private static func layout(anchors: [String]) throws -> H3PackedSequence {
        try H3Packing.buildPackedSequence(
            textTokenTags: [1, 1, 1],
            numLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            numAudioLatents: 20,
            patchSize: (1, 2, 2),
            keyframeAnchors: anchors
        )
    }

    @Test func conditionRowsAreOneFrameEach() throws {
        let one = try Self.layout(anchors: ["first"])
        #expect(one.numConditionVideoRows == Self.rowsPerFrame)
        let two = try Self.layout(anchors: ["first", "last"])
        #expect(two.numConditionVideoRows == 2 * Self.rowsPerFrame)
        #expect(two.sequenceLength == 3 + 2 * Self.rowsPerFrame + 40 + Self.latentFrames * Self.rowsPerFrame)
    }

    @Test func anchorsSitAtTheEndsOfTheVideoClock() throws {
        let layout = try Self.layout(anchors: ["first", "last"])
        let conditionStart = 3
        let firstTime = layout.positionIds[conditionStart, 0].item(Float.self)
        let lastTime = layout.positionIds[conditionStart + Self.rowsPerFrame, 0].item(Float.self)
        // "first" is pinned at the text length (where the media clock starts)...
        #expect(abs(firstTime - 3.0) < 1e-4)
        // ...and "last" one frame step before the end of the span.
        let span = H3Packing.temporalPositionSpan(numLatentFrames: Self.latentFrames)
        #expect(abs(lastTime - Float(3.0 + span - 5.0 / 3.0)) < 1e-3)
    }

    @Test func conditionRowsCarryTheVideoTagAndStayPinned() throws {
        let layout = try Self.layout(anchors: ["first"])
        for index in 3..<(3 + Self.rowsPerFrame) {
            #expect(layout.tokenTagValues[index] == H3Constants.videoTag)
        }
        let (timesteps, indices) = H3Packing.buildRowTimesteps(
            layout: layout, videoTimestep: 0.2, audioTimestep: 0.5,
            conditionVideoTimestep: 0.999, conditionAudioTimestep: 1.0)
        #expect(timesteps == [0.2, 0.5, 0.999])
        #expect(indices[3].item(Int32.self) == 2)   // condition row -> 0.999
        #expect(indices[0].item(Int32.self) == 0)   // text row inherits the video timestep
    }

    @Test func unknownAnchorIsRejected() throws {
        #expect(throws: H3Error.self) { try Self.layout(anchors: ["middle"]) }
    }

    @Test func patchifyRoundTrips() {
        let latents = MLXArray(0..<(1 * 24 * 4 * 8 * 6)).reshaped(1, 24, 4, 8, 6).asType(.float32)
        let rows = H3Packing.patchifyVideoLatents(latents, patchSize: (1, 2, 2))
        #expect(rows.shape == [4 * 4 * 3, 24 * 4])
        let restored = H3Packing.unpatchifyVideoTokens(
            rows, numLatentFrames: 4, latentHeight: 8, latentWidth: 6,
            channels: 24, patchSize: (1, 2, 2))
        #expect((restored - latents).abs().max().item(Float.self) == 0)
    }
}
