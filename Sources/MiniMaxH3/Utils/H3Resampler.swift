// H3Resampler.swift - torchaudio-compatible sinc resampling for reference soundtracks
// Copyright 2026 Vincent Gourbin
//
// A `ref2va` reference soundtrack is put on the audio VAE's 32 kHz by ONE `torchaudio.transforms.
// Resample` pass in the reference implementation, so this reproduces that transform rather than
// any convenient resampler: `sinc_interp_hann`, `lowpass_filter_width = 6`, `rolloff = 0.99`, the
// polyphase kernel built in float64 and applied in float32.
//
// The algorithm, from torchaudio's `_get_sinc_resample_kernel` / `_apply_sinc_resample_kernel`:
// after dividing both rates by their GCD, output `y[j]` is a sinc interpolation of `x` whose
// weights repeat with period `new_freq`, so the whole resample is one strided convolution against
// `new_freq` kernels of width `2 * width + orig_freq`, stride `orig_freq`, and the results are read
// out position-major / kernel-minor.

import Foundation

public enum H3Resampler {
    /// torchaudio's defaults, which are what the reference implementation used.
    static let lowpassFilterWidth = 6
    static let rolloff = 0.99

    static func greatestCommonDivisor(_ a: Int, _ b: Int) -> Int {
        var (x, y) = (abs(a), abs(b))
        while y != 0 { (x, y) = (y, x % y) }
        return x
    }

    /// The `new_freq` polyphase kernels, each `2 * width + orig_freq` taps, and `width`.
    /// `origFreq` / `newFreq` must already be divided by their GCD.
    static func sincKernels(origFreq: Int, newFreq: Int) -> (kernels: [[Float]], width: Int) {
        let baseFreq = Double(min(origFreq, newFreq)) * rolloff
        let filterWidth = Double(lowpassFilterWidth)
        let width = Int((filterWidth * Double(origFreq) / baseFreq).rounded(.up))
        let taps = 2 * width + origFreq

        var kernels = [[Float]](repeating: [], count: newFreq)
        for kernel in 0..<newFreq {
            var values = [Float](repeating: 0, count: taps)
            for tap in 0..<taps {
                // idx = (-width ..< width + orig_freq) / orig_freq ; t = -k/new_freq + idx
                let idx = Double(tap - width) / Double(origFreq)
                var t = (-Double(kernel) / Double(newFreq) + idx) * baseFreq
                t = min(max(t, -filterWidth), filterWidth)
                let window = pow(cos(t * Double.pi / filterWidth / 2.0), 2.0)
                t *= Double.pi
                let sinc = t == 0 ? 1.0 : sin(t) / t
                values[tap] = Float(sinc * window * (baseFreq / Double(origFreq)))
            }
            kernels[kernel] = values
        }
        return (kernels, width)
    }

    /// Resample one channel. `from`/`to` are the true rates; the GCD reduction happens here.
    public static func resample(_ samples: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
        guard sourceRate > 0, targetRate > 0 else {
            throw H3Error.invalidInput("Resampling needs positive rates, got \(sourceRate) -> \(targetRate).")
        }
        if sourceRate == targetRate { return samples }
        guard !samples.isEmpty else { return [] }

        let divisor = greatestCommonDivisor(sourceRate, targetRate)
        let origFreq = sourceRate / divisor
        let newFreq = targetRate / divisor
        let (kernels, width) = sincKernels(origFreq: origFreq, newFreq: newFreq)
        let taps = 2 * width + origFreq

        // `F.pad(waveform, (width, width + orig_freq))` — zero padding on both sides.
        let length = samples.count
        var padded = [Float](repeating: 0, count: length + 2 * width + origFreq)
        padded.replaceSubrange(width..<(width + length), with: samples)

        let positions = (padded.count - taps) / origFreq + 1
        let targetLength = Int((Double(newFreq) * Double(length) / Double(origFreq)).rounded(.up))
        var output = [Float](repeating: 0, count: positions * newFreq)

        // Kernels flattened into one buffer: the inner loop runs `positions * newFreq` times
        // (480 k for a 15 s clip at 44.1 kHz) and must not re-enter a closure on each pass.
        let flatKernels = kernels.flatMap { $0 }
        padded.withUnsafeBufferPointer { source in
            flatKernels.withUnsafeBufferPointer { weights in
                output.withUnsafeMutableBufferPointer { destination in
                    for position in 0..<positions {
                        let offset = position * origFreq
                        for kernel in 0..<newFreq {
                            var sum: Float = 0
                            let base = kernel * taps
                            for tap in 0..<taps { sum += source[offset + tap] * weights[base + tap] }
                            destination[position * newFreq + kernel] = sum
                        }
                    }
                }
            }
        }
        return Array(output.prefix(targetLength))
    }
}
