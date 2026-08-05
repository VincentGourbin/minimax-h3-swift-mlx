// PackingTests.swift - Geometry invariants from the reference implementation
// Copyright 2026 Vincent Gourbin

import MLX
import Testing
@testable import MiniMaxH3

@Suite struct GeometryTests {
    @Test func canvas16x9() throws {
        // 768 * 16/9 = 1365.33 -> area cap 768*1344 scales it back -> rounds to 1344x768.
        let (height, width) = try H3Geometry.resolveCanvasSize(aspectWidth: 16, aspectHeight: 9)
        #expect(height == 768 && width == 1344)
    }

    @Test func canvasSquare() throws {
        let (height, width) = try H3Geometry.resolveCanvasSize(aspectWidth: 1, aspectHeight: 1)
        #expect(height == 768 && width == 768)
    }

    @Test func frameAlignment() throws {
        #expect(try H3Geometry.alignNumFrames(124) == 124)  // 17*7+5 already aligned
        #expect(try H3Geometry.alignNumFrames(120) == 124)
        #expect(try H3Geometry.videoLatentNumFrames(124) == 37)  // 5*7+2
        #expect(H3Geometry.audioLatentNumFrames(124) == 207)  // round(124/24*40)
    }

    @Test func schedulerGrid() throws {
        let scheduler = H3Scheduler(shift: 12.0)
        try scheduler.setTimesteps(numInferenceSteps: 10)
        // N grid points -> N-1 model evaluations; terminal sigma is exactly 0.
        #expect(scheduler.sigmas.first == 1.0)
        #expect(scheduler.sigmas.last == 0.0)
        #expect(scheduler.timesteps.count == scheduler.sigmas.count - 1)
        // Shift compresses toward sigma = 1: second sigma stays high.
        #expect(scheduler.sigmas[1] > 0.9)
    }

    @Test func packedLayoutShape() throws {
        // Small synthetic case: 3 text tokens, 7 latent frames, 8x6 latent grid, 20 audio latents.
        let layout = try H3Packing.buildPackedSequence(
            textTokenTags: [1, 1, 1],
            numLatentFrames: 7,
            latentHeight: 8,
            latentWidth: 6,
            numAudioLatents: 20,
            patchSize: (1, 2, 2)
        )
        let rowsPerFrame = (8 / 2) * (6 / 2)
        #expect(layout.sequenceLength == 3 + 0 + 40 + 7 * rowsPerFrame)
        #expect(layout.numConditionVideoRows == 0)
        // Video temporal grid is non-uniform: frame 1 sits 5/3 after frame 0, frame 5 restarts the
        // (1,4,4,4,4) cycle: t(1) - t(0) = 5/3, t(2) - t(1) = 20/3.
        let videoStart = 3 + 40
        let t0 = layout.positionIds[videoStart, 0].item(Float.self)
        let t1 = layout.positionIds[videoStart + rowsPerFrame, 0].item(Float.self)
        let t2 = layout.positionIds[videoStart + 2 * rowsPerFrame, 0].item(Float.self)
        #expect(abs((t1 - t0) - 5.0 / 3.0) < 1e-4)
        #expect(abs((t2 - t1) - 20.0 / 3.0) < 1e-4)
        // Audio advances 1 unit per latent, starting at the text clock offset.
        let audioT0 = layout.positionIds[3, 0].item(Float.self)
        let audioT1 = layout.positionIds[4, 0].item(Float.self)
        #expect(audioT0 == 3.0 && audioT1 == 4.0)
    }

    @Test func rowTimestepPlan() throws {
        let layout = try H3Packing.buildPackedSequence(
            textTokenTags: [1, 1],
            numLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            numAudioLatents: 3,
            patchSize: (1, 2, 2)
        )
        let (timesteps, indices) = H3Packing.buildRowTimesteps(
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.5,
            conditionVideoTimestep: 0.999,
            conditionAudioTimestep: 1.0
        )
        // No conditioning rows -> two distinct values, sorted ascending.
        #expect(timesteps == [0.2, 0.5])
        // Text rows inherit the video timestep.
        #expect(indices[0].item(Int32.self) == 0)
        // Audio rows point at the audio timestep.
        #expect(indices[2].item(Int32.self) == 1)
    }
}

@Suite struct VAETests {
    @Test func blendNegativeAxis() throws {
        // Regression: blend() must accept the negative axes decodeClip uses (-1, -2).
        let a = MLXArray.ones([1, 3, 4, 8, 8])
        let b = MLXArray.zeros([1, 3, 4, 8, 8])
        let blended = H3VideoVAE.blend(a, b, extent: 4, axis: -1)
        #expect(blended.shape == [1, 3, 4, 8, 8])
        let blendedH = H3VideoVAE.blend(a, b, extent: 8, axis: -2)
        #expect(blendedH.shape == [1, 3, 4, 8, 8])
    }
}
