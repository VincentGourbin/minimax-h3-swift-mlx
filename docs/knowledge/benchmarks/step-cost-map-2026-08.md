---
okf_version: "0.1"
kind: benchmark
created: 2026-08-10
device: Apple M3 Max, 96 GB unified memory
---

# Where a denoising step goes — and what optimizing it could actually buy

Measured with `minimax-h3 bench --tokens N --quant qint8`, which times the hot-path primitives on
synthetic weights of the real shapes (hidden 5376, 56 heads × 128, ffn 14336, 50 blocks). No
checkpoint needed: the shapes decide the cost. Machine not strictly idle — treat ±15 % as noise.

## Numbers

| primitive | 8 998 tokens (576×384) | 39 930 tokens (1344×768) |
|---|---|---|
| attention (SDPA) | 13.3 s/step (18 %) | 287.5 s/step (49 %) |
| feed-forward | 27.8 s (37 %) | 184.0 s (31 %) |
| q/k/v projections | 14.1 s (19 %) | 79.3 s (13 %) |
| attention out projection | 5.4 s (7 %) | 30.6 s (5 %) |
| AdaLN gather + affine | 0.7 s | 3.1 s |
| rotary apply | 0.5 s | 2.4 s |
| RMSNorm | 0.3 s | 1.8 s |
| **sum of primitives** | **75.5 s** | **588 s** |
| **composed block × 50** | **75.8 s** | **1 261 s** |
| measured step | 162 s | 1 081 s |

## What this rules out

- **Attention is not 90 % of a step.** The earlier estimate (recorded in action plan 405) came
  from fitting two whole-run timings; measured directly, attention is **49 % of the block
  compute at the full canvas and 18 % at 576×384**. The feed-forward alone is comparable. So
  even a *free* attention would buy at most ~2× at 768p, not the 6× that plan assumed.
- **Transposed views into SDPA cost nothing**: feeding `(B, seq, H, D)` transposed (what
  production does) measured 268.6 ms versus 265.3 ms contiguous. There is no hidden copy to
  eliminate.
- **Bookkeeping is noise**: rotary, AdaLN gathers and RMSNorm together are ~1 % of a step.
  Hoisting the rotary table out of the loop, fusing the AdaLN gather, avoiding the rotary's
  concatenations — all of it is worth about one percent. Not where to spend effort.
- **Composition is not the problem at small canvases**: at 8 998 tokens the composed block costs
  exactly the sum of its parts (75.8 vs 75.5 s), so there is no allocator or glue overhead
  hiding between the primitives.

## The open question

At 8 998 tokens the blocks explain only **47 % of the measured step** (75.8 of 162 s). The gap is
~85 s and it does *not* scale with sequence length, so it is a fixed per-step cost outside the
transformer blocks. At the full canvas the same fixed term is a rounding error, which is why it
never showed up before.

Prime suspect: streaming 18.5 GB of distinct quantized weights every step, which the bench never
pays (it reuses one set of weights, kept hot). Supporting evidence: at 3.7k tokens the repo's own
quantization benchmark has int4 (9.5 GB of weights) *faster* than qint8 (18.5 GB) — 23.6 vs
28.0 s — where compute is nearly identical. Contradicting evidence: bf16 (62 GB) is only 11 %
slower than qint8 there, which pure streaming cannot explain.

**Decisive experiment, not yet run**: time the composed block over N *distinct* weight sets versus
one set reused N times, same compute either way. If distinct is much slower, the fixed cost is
memory traffic and more aggressive quantization attacks it directly.

## Ranked opportunities

1. **Settle the fixed ~85 s** (experiment above). At 576×384 it is *half the step* — a bigger
   lever than anything inside the blocks at that size, and int4 would halve it if it is streaming.
2. **Fewer sigma steps** — zero code risk, saving exactly linear in steps. 30 → 20 is −33 % of a
   run. Needs a quality comparison at equal seed, which is cheap to produce.
3. **Sparse attention** — up to ~2× at 768p by this measurement, not 6×. Still the largest
   single lever at the full canvas, but the payoff is half what plan 405 assumed, against a
   substantial implementation cost (the released MSA kernels are CUDA-only; only the algorithm
   transfers).
4. **Fuse the elementwise glue** — at 39 930 tokens the composed block costs 16 % more than the
   sum of its primitives; some of that is fusable. Modest and safe, but only worth it after 1-3.
5. **Nothing else in the block is worth touching**: see "bookkeeping is noise" above.
