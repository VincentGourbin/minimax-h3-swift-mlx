---
okf_version: "0.1"
kind: pitfall
created: 2026-08-05
---

# AVAssetWriter stalls when tracks are fed sequentially

**Symptom**: MP4 export hangs forever (0-byte file, all CoreMedia threads idle-waiting) for clips
longer than ~2-3 seconds, while short clips export fine. Per-frame appends are sub-millisecond,
but `isReadyForMoreMediaData` stops coming back after ~60 video frames.

**Cause (two layers)**:
1. Non-realtime `AVAssetWriterInput`s only pump reliably through `requestMediaDataWhenReady` —
   polling `isReadyForMoreMediaData` in a sleep loop is not a supported pattern.
2. The deeper one: a writer with several inputs **interleaves** samples across tracks and stops
   accepting video that runs further than its interleave window (~2-3 s) ahead of the audio
   track. Feeding *all* video before *any* audio therefore cross-blocks: video input refuses more
   frames until audio for the same time range arrives, which our sequential code only appended
   after the video finished. Short clips fit inside the window, masking the bug in smoke tests.

**Fix**: register both tracks' `requestMediaDataWhenReady` pumps **before awaiting either**, so the
muxer can interleave (`Sources/MiniMaxH3/Utils/VideoExporter.swift`, `TrackCompletion` latch).

**Also learned while debugging**: killed processes lose buffered stdout — `H3Debug.log` now
fflushes; and per-pixel Swift-array subscripts in frame-conversion loops cost ~100x vs
`withUnsafeBufferPointer` (kept the pointer version).

**Test**: `minimax-h3 mux` on a >5 s synthetic raw (124 frames) — the failing geometry.
