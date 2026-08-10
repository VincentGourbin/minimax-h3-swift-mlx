// KeyframeImageProcessor.swift - fl2va keyframe preparation + Qwen2-VL image preprocessing
// Copyright 2026 Vincent Gourbin
//
// Two stages, both ported for exactness:
//  1. Canvas preparation (diffusers `prepare_keyframe_image`): the first keyframe is *stretched*
//     onto the target canvas, a second one is cover-cropped (aspect-preserving max-scale resize
//     plus centre crop), both with PIL's LANCZOS resampling — reimplemented here with PIL's
//     exact fixed-point arithmetic (22-bit coefficients, two uint8 passes, horizontal first)
//     so canvas bytes match the reference bit for bit.
//  2. Qwen2-VL image processing: the canvas is already a multiple of 32 and inside the
//     processor's pixel bounds, so `smart_resize` is a no-op by construction — what remains is
//     rescale + normalize (mean/std 0.5), temporal replication x2, and the block-major patchify
//     into (gridH*gridW, 1536) rows whose per-patch layout is [channel][t][patchH][patchW].
//
// Image decode (JPEG/PNG -> RGB bytes) goes through ImageIO; codec-level byte differences vs
// PIL's decoders are possible and out of contract — parity harnesses feed decoded pixels.

import Foundation
import ImageIO
import MLX

/// An 8-bit interleaved RGB bitmap, the working currency of keyframe preparation.
public struct H3KeyframeImage: Sendable {
    public var width: Int
    public var height: Int
    /// Row-major interleaved RGB, `height * width * 3` bytes.
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    // MARK: - Loading

    /// Decode an image file to RGB with its EXIF orientation applied — PIL's `exif_transpose` +
    /// `convert("RGB")`.
    ///
    /// Alpha is *undone*, not composited: PIL keeps an RGBA image's colour channels untouched,
    /// while Core Graphics only hands back premultiplied pixels, which would darken every
    /// semi-transparent region and turn transparent ones black. We therefore rasterize with
    /// premultiplied alpha and divide it back out. A wrong canvas here would feed both the
    /// vision tower and the VAE conditioning rows.
    public static func load(from url: URL) throws -> H3KeyframeImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw H3Error.invalidInput("Cannot decode image at \(url.path)")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = properties?[kCGImagePropertyOrientation] as? UInt32 ?? 1

        let (width, height) = (cgImage.width, cgImage.height)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // The context keeps writing through this pointer during `draw`, so both calls must stay
        // inside the closure — passing `&rgba` to the initializer alone is undefined behaviour.
        var rasterized = true
        rgba.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else {
                rasterized = false
                return
            }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        guard rasterized else {
            throw H3Error.invalidInput("Cannot rasterize image at \(url.path)")
        }

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let alpha = Int(rgba[pixel * 4 + 3])
            for channel in 0..<3 {
                let value = Int(rgba[pixel * 4 + channel])
                // alpha == 0 carries no colour information; PIL would expose the raw channel,
                // which Core Graphics has already zeroed, so black is the only answer left.
                rgb[pixel * 3 + channel] = alpha == 0 || alpha == 255
                    ? UInt8(value)
                    : UInt8(min(255, (value * 255 + alpha / 2) / alpha))
            }
        }
        return H3KeyframeImage(width: width, height: height, pixels: rgb)
            .applyingEXIFOrientation(orientation)
    }

    /// Apply one of the 8 EXIF orientations by pixel permutation.
    func applyingEXIFOrientation(_ orientation: UInt32) -> H3KeyframeImage {
        guard (2...8).contains(orientation) else { return self }
        let swapsAxes = orientation >= 5
        let (outWidth, outHeight) = swapsAxes ? (height, width) : (width, height)
        var out = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<height {
            for x in 0..<width {
                let (dx, dy): (Int, Int)
                switch orientation {
                case 2: (dx, dy) = (width - 1 - x, y)                    // flip H
                case 3: (dx, dy) = (width - 1 - x, height - 1 - y)       // rotate 180
                case 4: (dx, dy) = (x, height - 1 - y)                   // flip V
                case 5: (dx, dy) = (y, x)                                // transpose
                case 6: (dx, dy) = (height - 1 - y, x)                   // rotate 90 CW
                case 7: (dx, dy) = (height - 1 - y, width - 1 - x)       // transverse
                default: (dx, dy) = (y, width - 1 - x)                   // 8: rotate 270 CW
                }
                let src = (y * width + x) * 3
                let dst = (dy * outWidth + dx) * 3
                out[dst] = pixels[src]
                out[dst + 1] = pixels[src + 1]
                out[dst + 2] = pixels[src + 2]
            }
        }
        return H3KeyframeImage(width: outWidth, height: outHeight, pixels: out)
    }

    // MARK: - Canvas preparation (diffusers `prepare_keyframe_image`)

    /// Put the keyframe onto the target canvas: stretched for the geometry anchor (first
    /// keyframe), cover-cropped for a follower. An image already at canvas size passes through
    /// untouched — no resampling pass, part of the reference contract.
    public func prepared(canvasWidth: Int, canvasHeight: Int, stretch: Bool) -> H3KeyframeImage {
        if width == canvasWidth && height == canvasHeight { return self }
        if stretch {
            return resizedLanczos(toWidth: canvasWidth, toHeight: canvasHeight)
        }
        let scale = max(Double(canvasWidth) / Double(width), Double(canvasHeight) / Double(height))
        // Python round() is banker's rounding — keep half-to-even.
        let resizedWidth = max(canvasWidth, Int((Double(width) * scale).rounded(.toNearestOrEven)))
        let resizedHeight = max(canvasHeight, Int((Double(height) * scale).rounded(.toNearestOrEven)))
        let left = max(0, (resizedWidth - canvasWidth) / 2)
        let top = max(0, (resizedHeight - canvasHeight) / 2)
        return resizedLanczos(toWidth: resizedWidth, toHeight: resizedHeight)
            .cropped(left: left, top: top, width: canvasWidth, height: canvasHeight)
    }

    func cropped(left: Int, top: Int, width cropWidth: Int, height cropHeight: Int) -> H3KeyframeImage {
        var out = [UInt8](repeating: 0, count: cropWidth * cropHeight * 3)
        for y in 0..<cropHeight {
            let src = ((top + y) * width + left) * 3
            let dst = y * cropWidth * 3
            out[dst..<(dst + cropWidth * 3)] = pixels[src..<(src + cropWidth * 3)]
        }
        return H3KeyframeImage(width: cropWidth, height: cropHeight, pixels: out)
    }

    // MARK: - PIL-exact LANCZOS resampling

    /// PIL's fixed-point precision for 8-bit resampling: 32 - 8 (pixel) - 2 (headroom).
    private static let precisionBits = 22

    private struct ResampleAxis {
        /// Per output index: first source index and tap count.
        var bounds: [(min: Int, count: Int)]
        /// Per output index: `count` fixed-point coefficients.
        var coefficients: [[Int32]]
    }

    /// Lanczos-3 kernel: sinc(x)·sinc(x/3) on |x| < 3.
    private static func lanczos(_ x: Double) -> Double {
        if x <= -3.0 || x >= 3.0 { return 0.0 }
        if x == 0.0 { return 1.0 }
        let pix = Double.pi * x
        return 3.0 * sin(pix) * sin(pix / 3.0) / (pix * pix)
    }

    /// PIL `precompute_coeffs` + `normalize_coeffs_8bpc`: double-precision windowed taps,
    /// normalized to sum 1, then rounded half-away-from-zero into 22-bit fixed point.
    private static func resampleAxis(inSize: Int, outSize: Int) -> ResampleAxis {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 3.0 * filterScale
        let inverseScale = 1.0 / filterScale

        var bounds = [(min: Int, count: Int)]()
        var coefficients = [[Int32]]()
        bounds.reserveCapacity(outSize)
        coefficients.reserveCapacity(outSize)
        for index in 0..<outSize {
            let center = (Double(index) + 0.5) * scale
            let low = max(0, Int(center - support + 0.5))
            let high = min(inSize, Int(center + support + 0.5))
            var taps = [Double](repeating: 0, count: high - low)
            var sum = 0.0
            for tap in 0..<taps.count {
                let weight = lanczos((Double(tap + low) - center + 0.5) * inverseScale) * inverseScale
                taps[tap] = weight
                sum += weight
            }
            let fixed = taps.map { weight -> Int32 in
                let normalized = sum != 0 ? weight / sum : weight
                let scaled = normalized * Double(1 << precisionBits)
                return Int32(normalized < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            bounds.append((low, high - low))
            coefficients.append(fixed)
        }
        return ResampleAxis(bounds: bounds, coefficients: coefficients)
    }

    private static func clip8(_ value: Int) -> UInt8 {
        if value >= (1 << (precisionBits + 8)) { return 255 }
        if value <= 0 { return 0 }
        return UInt8(value >> precisionBits)
    }

    /// Two-pass separable resize, horizontal then vertical, rounding to uint8 between passes —
    /// PIL's `ImagingResample` for 8-bit images, bit for bit.
    public func resizedLanczos(toWidth outWidth: Int, toHeight outHeight: Int) -> H3KeyframeImage {
        let rounding = 1 << (Self.precisionBits - 1)

        var horizontal = pixels
        var midWidth = width
        if outWidth != width {
            let axis = Self.resampleAxis(inSize: width, outSize: outWidth)
            var out = [UInt8](repeating: 0, count: outWidth * height * 3)
            for y in 0..<height {
                let rowBase = y * width * 3
                for x in 0..<outWidth {
                    let (low, count) = axis.bounds[x]
                    let taps = axis.coefficients[x]
                    var sums = (rounding, rounding, rounding)
                    for tap in 0..<count {
                        let src = rowBase + (low + tap) * 3
                        let weight = Int(taps[tap])
                        sums.0 += weight * Int(horizontal[src])
                        sums.1 += weight * Int(horizontal[src + 1])
                        sums.2 += weight * Int(horizontal[src + 2])
                    }
                    let dst = (y * outWidth + x) * 3
                    out[dst] = Self.clip8(sums.0)
                    out[dst + 1] = Self.clip8(sums.1)
                    out[dst + 2] = Self.clip8(sums.2)
                }
            }
            horizontal = out
            midWidth = outWidth
        }

        guard outHeight != height else {
            return H3KeyframeImage(width: midWidth, height: height, pixels: horizontal)
        }
        let axis = Self.resampleAxis(inSize: height, outSize: outHeight)
        var out = [UInt8](repeating: 0, count: midWidth * outHeight * 3)
        for y in 0..<outHeight {
            let (low, count) = axis.bounds[y]
            let taps = axis.coefficients[y]
            for x in 0..<midWidth {
                var sums = (rounding, rounding, rounding)
                for tap in 0..<count {
                    let src = ((low + tap) * midWidth + x) * 3
                    let weight = Int(taps[tap])
                    sums.0 += weight * Int(horizontal[src])
                    sums.1 += weight * Int(horizontal[src + 1])
                    sums.2 += weight * Int(horizontal[src + 2])
                }
                let dst = (y * midWidth + x) * 3
                out[dst] = Self.clip8(sums.0)
                out[dst + 1] = Self.clip8(sums.1)
                out[dst + 2] = Self.clip8(sums.2)
            }
        }
        return H3KeyframeImage(width: midWidth, height: outHeight, pixels: out)
    }

    // MARK: - Qwen2-VL patchify

    /// Rescale + normalize (mean/std 0.5), replicate the single frame over the temporal patch
    /// axis, and flatten to vision-tower rows in the processor's block-major order. Per-patch
    /// feature layout: [channel][t][patchH][patchW] -> 3*2*16*16 = 1536.
    /// The canvas must already satisfy `smart_resize` (multiple of 32, pixel count in bounds).
    public func visionPatches(
        patchSize: Int = 16, mergeSize: Int = 2, temporalPatchSize: Int = 2
    ) -> (patches: MLXArray, gridH: Int, gridW: Int) {
        let factor = patchSize * mergeSize
        precondition(
            width % factor == 0 && height % factor == 0,
            "Canvas \(width)x\(height) must be a multiple of \(factor)")
        let (gridH, gridW) = (height / patchSize, width / patchSize)
        let (blocksH, blocksW) = (gridH / mergeSize, gridW / mergeSize)

        var x = MLXArray(pixels, [height, width, 3]).asType(.float32)
        x = (x / 255.0 - 0.5) / 0.5
        x = x.transposed(2, 0, 1)  // (C, H, W)
        x = x.reshaped(3, blocksH, mergeSize, patchSize, blocksW, mergeSize, patchSize)
        // -> [hBlock, wBlock, mergeH, mergeW, C, patchH, patchW], the block-major token order.
        x = x.transposed(1, 4, 2, 5, 0, 3, 6)
        x = x.expandedDimensions(axis: 5)  // temporal axis between C and patchH
        x = broadcast(
            x,
            to: [blocksH, blocksW, mergeSize, mergeSize, 3, temporalPatchSize, patchSize, patchSize])
        let patches = x.reshaped(gridH * gridW, 3 * temporalPatchSize * patchSize * patchSize)
        return (patches, gridH, gridW)
    }
}
