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

## The clean reference (measured after the fact — read this first)

`generate -W 576 -H 384 -f 124 -s 7 --profile` (8 998 tokens, qint8 both components, only light
background load): **62.9 s/step, σ 4.6 s** over 6 steps (min 55.1, max 69.5).

| | s/step |
|---|---|
| clean measurement | **62.9** |
| this document's bench prediction | 75.8 (overestimates by 21 %) |
| the figure the map was originally built against | 162 (inflated **2.6×** by contention) |

So the bench slightly *over*states the real cost, and the "gap" analysed below never existed.
Phase split of that short run: denoising 68 % (GPU 74 %), VAE decode 16 % (GPU 49 %), transformer
load 7.5 %, peak MLX 34.4 GB.

**GPU utilization**: 74 % during denoising at this size. The missing quarter is launch gaps and
unfused elementwise work between kernels.

### Second clean point — and it moves the conclusion

`generate -W 576 -H 384 -f 345 -s 30 --profile` (24 005 tokens, the longest clip H3 accepts):
**401 s/step, σ 16.5 s** over 29 steps, denoising 97 % of the run, **GPU 92 %**, peak MLX 34.4 GB.

Two things follow, and both correct what is written above:

1. **The elementwise-fusion lever shrinks as the canvas grows.** GPU occupancy goes 74 % → 92 %
   between 9 k and 24 k tokens: the kernels get big enough to hide the launch gaps by themselves.
   Fusing the block's glue is worth ~18 % at 576×384/124f and only ~8 % at the long clip. It is a
   *small-canvas* optimization, not a general one.
2. **Cost grows as roughly tokens^1.9 in this range** (×2.67 tokens → ×6.38 time), close to
   quadratic and much steeper than "quadratic attention plus linear everything else" predicts
   (that model said 209 s; reality said 401 s). Estimates made from the primitive breakdown
   therefore *understate* large runs by about 2×. Use the measured exponent for projections until
   the discrepancy is understood — attention's real share at large sizes is probably higher than
   the bench's 49 %.

## The gap that wasn't

At 8 998 tokens the blocks first appeared to explain only 47 % of the measured step (75.8 of
162 s), suggesting a fixed ~85 s living outside them. The obvious suspect was streaming 18.5 GB
of distinct quantized weights per step, which the bench never pays because it reuses one hot set.

**Measured, and refuted** (`bench --weight-sets 12`): running the same arithmetic over twelve
*distinct* weight sets costs **0.5 % more** than reusing one — 0.01 s per block, about **1 s per
step**. Weight traffic is not a cost here. (Consistent with the older quantization benchmark's
awkward pair of facts: int4 beat qint8 by 4.4 s at 3.7k tokens while bf16, three times heavier
than qint8, was only 11 % slower. Neither is a streaming signature.)

What remains is not a mystery, it is **measurement uncertainty on the reference figures**. The
162 s/step came from the 2CV run while Teams and a Parallels VM were competing for the machine —
the same run's early steps measured 270 s. The bench itself varies ~30 % between compositions of
the same work. So the "gap" is within the noise of both sides, and there is nothing to chase.

**Before optimizing anything, produce one clean reference**: a short profiled run on a strictly
idle machine (`generate --profile`), which the repo's own benchmark methodology already demands.
Optimizing against a number measured under contention is how phantom targets get chased.

## Ranked opportunities

1. **Fewer sigma steps** — zero code risk, saving exactly linear in steps: 30 → 20 is −33 % of a
   run. Needs only a same-seed quality comparison, which the CLI can already produce. Nothing
   below is this cheap.
2. **Close the GPU-utilization gap, at small canvases only** — 74 % occupancy at 8 998 tokens
   means ~18 % is recoverable by fusing the block's elementwise glue (AdaLN affine, gates,
   residuals). But occupancy is already 92 % at 24 005 tokens, so the same work buys ~8 % there
   and less beyond. Safe and bounded, but do not expect it to help the runs that hurt most.
3. **Sparse attention** — up to ~2× at 768p by this measurement, not 6×, and nothing at small
   canvases (18 % of block compute there). Still the largest single lever at the full canvas,
   but against a substantial implementation cost: the released MSA kernels are CUDA-only, so
   only the algorithm transfers.
4. **Nothing else in the block is worth touching**: rotary, AdaLN gathers and RMSNorm are ~1 %,
   transposed views into SDPA cost nothing, weight traffic costs nothing (all measured above).

**And a standing rule, learned the hard way here**: measure the reference on an otherwise idle
machine before optimizing against it. The first version of this document chased an 85 s phantom
that was pure contention in the reference figure.
