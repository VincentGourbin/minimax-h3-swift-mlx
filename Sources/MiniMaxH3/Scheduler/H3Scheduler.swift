// H3Scheduler.swift - Rectified-flow Euler scheduler of MiniMax-H3 (port of scheduling_minimax_h3.py)
// Copyright 2026 Vincent Gourbin
//
// Three H3 specifics vs a standard flow-match Euler scheduler:
//  1. The velocity is data-ward: x0 = x_t + sigma * v (plus sign).
//  2. Timesteps are t = 1 - sigma in [0, 1], t = 1 meaning clean, consumed unscaled by the AdaLN.
//  3. The sigma grid is linspace(1, 0, numInferenceSteps) — the terminal 0 counts toward the
//     requested step count, so N grid points drive N-1 model evaluations.
// Eta is 0: no noise is ever re-injected. H3 runs two instances per request:
// shift 12.0 for video latents, shift 3.0 for audio latents.

import Foundation
import MLX

public final class H3Scheduler {
    public private(set) var shift: Float
    public private(set) var sigmas: [Float] = []
    /// Timesteps `1 - sigmas[..<(count-1)]`; one model evaluation each.
    public private(set) var timesteps: [Float] = []
    private var stepIndex = 0

    public init(shift: Float) {
        precondition(shift > 0, "`shift` must be positive, got \(shift)")
        self.shift = shift
    }

    /// Override the sigma shift; call before `setTimesteps`. Exposed per request upstream as
    /// `flow_shift` (video) / `audio_flow_shift` (audio).
    public func setShift(_ shift: Float) {
        precondition(shift > 0, "`shift` must be positive, got \(shift)")
        self.shift = shift
    }

    /// Build the schedule: linspace(1, 0, steps) through the exponential shift
    /// `sigma' = s*sigma / (1 + (s-1)*sigma)`, consecutive float32 duplicates collapsed.
    public func setTimesteps(numInferenceSteps: Int) throws {
        guard numInferenceSteps >= 2 else {
            throw H3Error.invalidConfiguration("`numInferenceSteps` must be >= 2, got \(numInferenceSteps).")
        }
        var grid = [Float]()
        grid.reserveCapacity(numInferenceSteps)
        let last = numInferenceSteps - 1
        for index in 0...last {
            // torch.linspace(1, 0, N) in float32.
            let base = Float(1.0 - Double(index) / Double(last))
            let shifted = shift * base / (1 + (shift - 1) * base)
            if grid.last != shifted { grid.append(shifted) }
        }
        sigmas = grid
        timesteps = grid.dropLast().map { 1.0 - $0 }
        stepIndex = 0
    }

    /// Rectified-flow forward process in H3's t convention: `x_t = t*x0 + (1-t)*noise`.
    /// Used to noise conditioning anchors, where `t` is the noise-aug level (not a schedule entry).
    public func scaleNoise(sample: MLXArray, timestep: Float, noise: MLXArray) -> MLXArray {
        timestep * sample + (1.0 - timestep) * noise
    }

    /// One Euler step. `modelOutput` is the data-ward velocity at `timestep`; sample is x_t.
    /// `x0 = x_t + (1-t)*v`, then `x_next = r*x_t + (1-r)*x0` with `r = sigma_next / sigma`,
    /// evaluated in float32.
    public func step(modelOutput: MLXArray, timestep: Float, sample: MLXArray) throws -> MLXArray {
        guard stepIndex + 1 < sigmas.count else {
            throw H3Error.generationFailed("Scheduler stepped past the end of its schedule.")
        }
        // Sigma for x0 is recovered from the timestep the transformer was conditioned on; the
        // Euler ratio uses the sigma grid. The reference keeps the two sources apart (the float32
        // round trip 1 - (1 - sigma) is not exact below 0.5).
        let sigmaFromTimestep = 1.0 - timestep
        let compute = sample.asType(.float32)
        let denoised = compute + sigmaFromTimestep * modelOutput.asType(.float32)

        let ratio = sigmas[stepIndex + 1] / sigmas[stepIndex]
        let previous = ratio * compute + (1.0 - ratio) * denoised
        stepIndex += 1
        return previous.asType(sample.dtype)
    }
}
