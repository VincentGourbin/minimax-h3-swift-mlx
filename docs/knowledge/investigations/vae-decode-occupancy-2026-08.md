---
okf_version: "0.1"
kind: investigation
created: 2026-08-13
updated: 2026-08-14
status: closed — every hypothesis refuted, the decode is GPU-saturated
device: Apple M3 Max, 96 GB unified memory
---

# The video VAE decode does NOT leave the GPU half idle

Issue #4. Every profiled run reports the same pair of numbers for the decode phase: **GPU 49 %,
CPU 3.4 %**, against **GPU 74 %, CPU 1.4 %** for denoising on the same run. This note is the
investigation of that pair. The first half below is what the geometry and the arithmetic said
before any GPU was occupied; the results section at the end is what the measurements said, and
they refute the premise — including the 49 % itself, which reads 99–100 % when sampled directly.

## The decode workload is N copies of one shape

Derived from `H3VideoVAE.decode` / `decodeClip` (temporal chunking `clip_length 17`, `token_drop 3`;
spatial tiling 256 px, min overlap 64, blended):

| run | latents (T,H,W) | chunks | tile grid | decoder passes | shape of every pass |
|---|---|---|---|---|---|
| 576×384 / 124f (reference) | 37 × 24 × 36 | 7 | 2 × 3 | **42** | 1797 tokens |
| 576×384 / 345f (14.4 s) | 102 × 24 × 36 | 20 | 2 × 3 | **120** | 1797 tokens |
| 512×288 / 22f (smoke) | 7 × 18 × 32 | 1 | 2 × 3 | **6** | 1797 tokens |
| 1344×768 / 124f (768p) | 37 × 48 × 84 | 7 | 4 × 7 | **196** | 1797 tokens |

**Every decoder pass, in every geometry, has exactly the same shape**: 7 latent frames × a 16×16
latent tile (= a 256×256 px tile) + 4 register tokens + 1 cls = 1797 tokens, 36 blocks, dim 2048.
Two properties make that a lever rather than a curiosity:

- **Uniformity is guaranteed, not lucky.** `splitTiles` distributes its slack in whole
  `spatialCompressionRatio` steps and the canvas is a multiple of 32, so the tiles always come out
  at exactly `tileSampleMinSize` and cover the span exactly — there is no ragged edge tile. Only
  the last temporal chunk can be short, and at the standard geometries it is not.
- **The passes are independent.** Attention never crosses a tile or a chunk; the blends happen
  afterwards, on the decoded pixels. So batching them along the batch axis is *exact*, not an
  approximation — unlike enlarging the tiles, which changes the blend geometry and therefore the
  output.

Today they run strictly one at a time, with an `eval()` barrier after each
(`H3VideoVAEDecoder.swift:440`).

## What the arithmetic rate says

Decoder: 2.417 B parameters (4.83 GB fp16, matching the 4 628 MB the profiler reports for the
loaded VAE). One pass = 2 × 1797 × 2.417 G + attention = **9.65 TFLOP**. The reference decode is
42 passes = **405 TFLOP in 87.2 s = 4.65 TFLOP/s** (2.08 s per pass).

That is the number to attack. It is *not* bandwidth-bound: 1 800 FLOP per weight byte, two orders
of magnitude above the machine's balance, and total weight traffic over the phase (42 × 4.83 GB =
203 GB) accounts for well under a second. It is not CPU-bound either: the profiler's CPU % is
process CPU time over wall time, so 3.4 % means the whole decode spends **3 CPU-seconds out of
87** — graph building and the rotary tables cannot be starving anything.

So the missing time is inside the GPU work itself: GEMMs of 1797 rows that may simply not saturate
the machine, plus ~54 memory-bound elementwise kernels per block (the fp32 casts around every
norm, the two per-head q/k RMSNorms, the rotary slice/concat pairs, four transposes, the
LayerScale gates) against 6 matmuls that carry all the FLOPs.

**Caveat on the 49 % itself**: it is `Device Utilization %` sampled from the AGX driver — a coarse
device-wide duty cycle. Denoising reads 74 % while it is certainly closer to saturated. Read the
pair as "decode leaves materially more headroom than denoise", not as "51 % of the GPU is idle".
The honest target is the 4.65 TFLOP/s, measured against what the same GEMMs sustain when they are
made bigger.

## Ranked hypotheses

1. **The GEMMs are too small.** 1797 rows per matmul, 42–196 times. Batching B passes into one
   call multiplies the row count by B, divides the kernel launches and the barriers by B, and
   costs one FFN intermediate per extra pass in memory: 1797 × 8192 × 2 B ≈ **29 MB per pass**
   (B = 6 → 0.2 GB, B = 42 → 1.5 GB), against a run peak of 34.4 GB set by the transformer stage
   and a decode phase that currently peaks at 5.0 GB. Exact math. This is hypothesis 1 and the
   implementation is ~30 lines in `decodeClip` (batch the tile loop; then, optionally, the chunks).
2. **The per-pass `eval()` barrier.** Subsumed by (1) if batching wins, but worth its own
   measurement: it says whether the loop is serialized by the barrier or by the work.
3. **The elementwise glue.** 54 of 60 kernels per block are memory-bound; `MLX.compile` fuses
   exactly this class. The transformer campaign found compile roughly neutral there — but the
   decoder's arithmetic intensity per token is lower, so the verdict does not transfer.
4. **The rotary table.** Rebuilt from a CPU triple loop on *every* pass although it is identical
   for all of them (same shape, always). Bounded by the 3 CPU-seconds above, so this is hygiene,
   not a lever — but it is free.
5. **Bigger tiles.** Changes the blend geometry, so it changes the output: a fidelity question,
   not a free win. Last.

## Measurement protocol (needs an idle GPU)

`minimax-h3 bench-decode --batches 1,2,3,6` — synthetic weights at the real shapes, no checkpoint,
~1 minute. It reports, per batch size, s/call, s/pass, the sustained TFLOP/s and the extrapolated
phase time; then the barrier's own cost (6 passes with an `eval()` each vs one at the end) and the
CPU cost of the rotary rebuild. Add `--batches 1,6,42` (~4 min) for the aggressive variant and
`--compile-blocks` for hypothesis 3.

Decision rule: if s/pass falls with batch size, implement batching in `decodeClip` (tiles first,
then chunks, keeping a batch cap as a memory knob) and validate bit-exactness against a stored
decode before/after. If it does not, the decoder is already running at the machine's rate for
these shapes and issue #4 closes as "no headroom without a fused kernel", like the sparse one.

Standing method note: idle machine, control sample back to back — that discipline has already
reversed two conclusions in this repo.

---

# Results (2026-08-14) — every hypothesis refuted

Measured with `minimax-h3 bench-decode`, cold machine, 5-minute cooldown before each point (see
the protocol caveat below — without it these numbers are worthless).

## Batching the passes: refuted, it makes things worse

| batch | s/call | s/pass | TFLOP/s | vs batch 1 |
|---|---|---|---|---|
| 1 | 1.143 | 1.143 | 8.43 | 1.00× |
| 2 | 2.244 | 1.122 | 8.59 | 1.02× |
| 3 | 3.621 | 1.207 | 7.99 | 0.95× |
| 6 | 8.335 | 1.389 | 6.94 | **0.82×** |

A 1797-token GEMM already saturates the machine; fusing passes only adds memory traffic. The
whole premise of hypothesis 1 — "the GEMMs are too small" — is wrong.

## The rest of the suspects

- **Per-pass `eval()` barrier**: −0.4 %, +2.8 %, −19.9 % across runs — i.e. noise, no signal.
- **Rotary table rebuilt per pass**: 0.1 ms, so 4 ms over a whole decode. Hygiene at most.
- **`MLX.compile` on the blocks** (A/B/B/A, cooldown before each point): compiled 1.116 / 1.135 s
  vs eager 1.138 / 1.164 s → −2.2 % against a ±2 % intra-variant spread. Neutral, same verdict as
  the transformer campaign.
- **Everything outside the 36 blocks**, timed at the real shapes: proj_in 0.04 s, norm_out +
  proj_out 0.14 s, the 8-D pixel-shuffle permute 0.01 s, tile stitching 0.04 s, temporal blends
  0.01 s, fp32 post-processing 0.02 s — **0.26 s out of 87**, 0.3 % of the phase.
- **The 49 % itself**: sampling `Device Utilization %` once a second during a real checkpoint
  decode gives **99–100 %** throughout. The profiler's phase average is not measuring headroom.

## Why the bench said 48 s and production says 87 s

Not missing work — the GPU's clock state. The same binary, same shapes, same machine, run at
different moments: **1.14 → 11.1 → 2.25 → 1.70 s/pass**. After five idle minutes it came back to
**1.139**, three thousandths from the very first measurement. A short burst on a cold GPU sustains
8.4 TFLOP/s; an 80-second phase — which is by definition sustained load — runs around 5. The
87.2 s the profiler reports for the decode phase is the sustained truth; the bench's 48 s is the
burst rate, and no amount of orchestration reaches it.

Consequence for every future bench in this repo: see
[gpu-burst-vs-sustained](../pitfalls/gpu-burst-vs-sustained.md).

## Verdict

The video VAE decode is **GPU-saturated and running at the machine's sustained rate**. There is no
orchestration headroom: not in batching, not in the barrier, not in kernel fusion, not in the glue.
Issue #4 closes the way #2 did — the premise was an artifact of the metric.

The only lever left is arithmetic, not orchestration: **the 256 px tiling decodes 1.78× more pixels
than the canvas has** at 576×384 (6 tiles × 256² = 393 k px for a 221 k px frame). Decoding a chunk
untiled would cut the linear work 44 % while raising attention (6048 tokens vs 6 × 1797) — net
**−31 % of the decode work** at this canvas. But it inverts at 768p, where an untiled chunk is
28 224 tokens and attention explodes (235 TFLOP against 90 for the tiled path), and it changes the
output, since the released frames are the blended-tile ones and that is what parity was established
against. For 16 % of a short run and nothing at large canvases, it does not pay. Recorded, not built.
