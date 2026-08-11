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

1. **Get one clean reference step** (idle machine, `--profile`). Cheap, and everything below is
   judged against it. Two of this document's three surprises came from comparing against numbers
   measured under load.
2. **Fewer sigma steps** — zero code risk, saving exactly linear in steps: 30 → 20 is −33 % of a
   run. Needs only a same-seed quality comparison, which the CLI can already produce.
3. **Sparse attention** — up to ~2× at 768p by this measurement, not 6×, and nothing at small
   canvases (18 % of block compute there). Still the largest single lever at the full canvas,
   but against a substantial implementation cost: the released MSA kernels are CUDA-only, so
   only the algorithm transfers.
4. **Fuse the elementwise glue** — at 39 930 tokens the composed block costs 16 % more than the
   sum of its primitives; part of that is fusable. Modest and safe, worth it only after 1-3.
5. **Nothing else in the block is worth touching**: rotary, AdaLN gathers and RMSNorm are ~1 %,
   transposed views into SDPA cost nothing, weight traffic costs nothing.
