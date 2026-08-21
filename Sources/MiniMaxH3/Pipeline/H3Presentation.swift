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

    init(tokenIds: [Int32], tokenTags: [Int32], mmTokenTypes: [Int32], imageGrids: [(h: Int, w: Int)]) {
        self.tokenIds = tokenIds
        self.tokenTags = tokenTags
        self.mmTokenTypes = mmTokenTypes
        self.imageGrids = imageGrids
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

// MARK: - ref2va

/// What one `ref2va` reference contributes to the presentation, in request order.
///
/// Audio contributes a label and nothing else — a waveform never reaches the conditioner.
public enum H3PresentationReference: Sendable {
    /// An image reference: one vision block at its (gridH, gridW).
    case image(gridH: Int, gridW: Int)
    /// A video reference: one timestamped vision block per merged frame pair, all at the same grid.
    case video(gridH: Int, gridW: Int, blockTimestamps: [Double])
    /// A standalone audio reference.
    case audio
    /// A video reference that carries a soundtrack: it is labelled `"<Audio j>: "` BEFORE
    /// `"<Video k>: "`, mirroring the order its rows are packed in.
    case videoWithAudio(gridH: Int, gridW: Int, blockTimestamps: [Double])
}

extension H3Presentation {
    /// Build MiniMax-H3's `ref2va` presentation: one label per reference, numbered PER MODALITY in
    /// request order, then the prompt verbatim.
    ///
    /// - `"<Picture i>: "` + a vision block for an image;
    /// - `"<Audio j>: "` alone for an audio reference — and also, first, for a video reference that
    ///   carries a soundtrack, which is what makes the audio counter run across both;
    /// - `"<Video k>: "` then, per merged frame pair, `"<t seconds>"` + a vision block.
    public static func ref2va(
        prompt: String,
        references: [H3PresentationReference],
        mergeSize: Int = 2,
        tokenizer: any Tokenizer
    ) throws -> H3Presentation {
        guard let visionStart = tokenizer.convertTokenToId("<|vision_start|>"),
              let visionEnd = tokenizer.convertTokenToId("<|vision_end|>"),
              let imagePad = tokenizer.convertTokenToId("<|image_pad|>"),
              let videoPad = tokenizer.convertTokenToId("<|video_pad|>")
        else {
            throw H3Error.invalidInput("Tokenizer lacks the Qwen vision special tokens.")
        }

        var ids = [Int32]()
        var tags = [Int32]()
        var types = [Int32]()
        var grids = [(h: Int, w: Int)]()

        func text(_ value: String) {
            let encoded = tokenizer.encode(text: value, addSpecialTokens: false)
            ids += encoded.map(Int32.init)
            tags += [Int32](repeating: H3Constants.textTag, count: encoded.count)
            types += [Int32](repeating: 0, count: encoded.count)
        }

        /// A vision block: `<|vision_start|>` + one pad per merged token + `<|vision_end|>`.
        /// H3 tags the whole block as video rows; the mrope token types mark only the pads.
        func vision(pad: Int, type: Int32, gridH: Int, gridW: Int) {
            let merged = (gridH / mergeSize) * (gridW / mergeSize)
            ids.append(Int32(visionStart))
            ids += [Int32](repeating: Int32(pad), count: merged)
            ids.append(Int32(visionEnd))
            tags += [Int32](repeating: H3Constants.videoTag, count: merged + 2)
            types.append(0)
            types += [Int32](repeating: type, count: merged)
            types.append(0)
            grids.append((gridH, gridW))
        }

        var pictures = 0
        var videos = 0
        var audios = 0
        for reference in references {
            switch reference {
            case .audio:
                audios += 1
                text("<Audio \(audios)>: ")
            case .image(let gridH, let gridW):
                pictures += 1
                text("<Picture \(pictures)>: ")
                vision(pad: imagePad, type: 1, gridH: gridH, gridW: gridW)
            case .video(let gridH, let gridW, let timestamps),
                 .videoWithAudio(let gridH, let gridW, let timestamps):
                if case .videoWithAudio = reference {
                    audios += 1
                    text("<Audio \(audios)>: ")
                }
                videos += 1
                text("<Video \(videos)>: ")
                for timestamp in timestamps {
                    text(H3ReferenceVideoProcessor.timestampLabel(timestamp))
                    vision(pad: videoPad, type: 2, gridH: gridH, gridW: gridW)
                }
            }
        }

        let promptIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
        ids += promptIds.map(Int32.init)
        tags += [Int32](repeating: H3Constants.textTag, count: promptIds.count)
        types += [Int32](repeating: 0, count: promptIds.count)

        guard !ids.isEmpty else { throw H3Error.invalidInput("Empty presentation.") }
        return H3Presentation(tokenIds: ids, tokenTags: tags, mmTokenTypes: types, imageGrids: grids)
    }
}
