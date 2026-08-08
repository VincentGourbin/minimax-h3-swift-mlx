// H3Presentation.swift - MiniMax-H3's token-level presentation of a request
// Copyright 2026 Vincent Gourbin
//
// t2va: the verbatim prompt, no chat template, no special tokens, every row tagged TEXT.
// fl2va: per keyframe, a `"<Picture i>: "` label (TEXT rows) then a vision block
// `<|vision_start|>` + one `<|image_pad|>` per merged vision token + `<|vision_end|>` (VIDEO
// rows — H3's AdaLN keys off that tag), then the verbatim prompt. Note the two taggings differ:
// H3's row tags mark the whole vision block as video, while the mrope token types (processor
// convention) mark only the `<|image_pad|>` tokens as image — the delimiters take ordinary
// sequential text positions.

import Foundation
import Tokenizers

public struct H3Presentation {
    public let tokenIds: [Int32]
    /// H3 AdaLN row tags: text 1, vision blocks 0 (video).
    public let tokenTags: [Int32]
    /// mrope token types: 0 text, 1 image — feeds `Qwen3VLMultimodalLayout`.
    public let mmTokenTypes: [Int32]
    /// Per image, the (gridH, gridW) patch grid before spatial merge.
    public let imageGrids: [(h: Int, w: Int)]

    /// The multimodal rotary layout of this presentation (nil for pure text).
    public func multimodalLayout(mergeSize: Int = 2) throws -> Qwen3VLMultimodalLayout? {
        guard !imageGrids.isEmpty else { return nil }
        return try Qwen3VLMultimodalLayout(
            mmTokenTypes: mmTokenTypes, imageGrids: imageGrids, mergeSize: mergeSize)
    }

    /// Build the presentation. `imageGrids` empty -> t2va (verbatim prompt).
    public init(
        prompt: String,
        imageGrids: [(h: Int, w: Int)] = [],
        mergeSize: Int = 2,
        tokenizer: any Tokenizer
    ) throws {
        var ids = [Int32]()
        var tags = [Int32]()
        var types = [Int32]()

        if !imageGrids.isEmpty {
            guard let visionStart = tokenizer.convertTokenToId("<|vision_start|>"),
                  let visionEnd = tokenizer.convertTokenToId("<|vision_end|>"),
                  let imagePad = tokenizer.convertTokenToId("<|image_pad|>")
            else {
                throw H3Error.invalidInput("Tokenizer lacks the Qwen vision special tokens.")
            }
            for (index, grid) in imageGrids.enumerated() {
                let labelIds = tokenizer.encode(
                    text: "<Picture \(index + 1)>: ", addSpecialTokens: false)
                let mergedTokens = (grid.h / mergeSize) * (grid.w / mergeSize)
                ids += labelIds.map(Int32.init)
                tags += [Int32](repeating: H3Constants.textTag, count: labelIds.count)
                types += [Int32](repeating: 0, count: labelIds.count)

                ids.append(Int32(visionStart))
                ids += [Int32](repeating: Int32(imagePad), count: mergedTokens)
                ids.append(Int32(visionEnd))
                tags += [Int32](repeating: H3Constants.videoTag, count: mergedTokens + 2)
                types.append(0)
                types += [Int32](repeating: 1, count: mergedTokens)
                types.append(0)
            }
        }
        let promptIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
        ids += promptIds.map(Int32.init)
        tags += [Int32](repeating: H3Constants.textTag, count: promptIds.count)
        types += [Int32](repeating: 0, count: promptIds.count)

        guard !ids.isEmpty else { throw H3Error.invalidInput("Empty presentation.") }
        tokenIds = ids
        tokenTags = tags
        mmTokenTypes = types
        self.imageGrids = imageGrids
    }
}
