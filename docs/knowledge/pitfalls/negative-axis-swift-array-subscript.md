---
okf_version: "0.1"
kind: pitfall
created: 2026-08-05
---

# Negative MLX axes crash Swift Array subscripts

**Symptom**: `Swift runtime failure: Index out of range` deep in a tensor helper, only on some
input sizes (here: video VAE spatial tiling, first triggered at 512 px width = 2 tiles, while all
smoke tests at ≤256 px ran single-tile and never hit the code path).

**Cause**: functions that accept an MLX-style axis (`-1`, `-2`) but also build a Swift
`[Int]` shape array and index it with that axis: `shape[axis] = extent` traps for negative axis.
MLX ops accept negative axes; Swift arrays do not.

**Fix**: normalize first — `let axis = axis < 0 ? array.ndim + axis : axis` — at the top of any
helper that mixes MLX axes with Swift array indexing. Regression test: `VAETests.blendNegativeAxis`.

**Lesson**: shape-level smoke tests must cover the branchy geometry (tiling, chunk overlap,
padding) — pick sizes that trigger ≥2 tiles and ≥2 temporal chunks, not the minimal case.
