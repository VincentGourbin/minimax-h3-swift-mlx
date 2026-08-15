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

## MLX.compile on the production blocks: bit-exact, roughly neutral

Tried (opt-in `generate --compile-blocks`, kept in the tree): each block's forward wrapped in
`MLX.compile`, weights captured as constants. The parity harness shows the compiled path is
**bit-exact** against eager (max|Δ| = 0.0 on both heads), and `temb`'s row count changing between
step 1 (both schedulers at σ=1 collapse to one distinct timestep) and step 2 forces one
recompilation — after that the signature is stable.

Measured back to back on the same (contended) machine at 8 998 tokens: compiled 109.5 s/step
average vs eager control 114.0, steady-state minima 74 vs 80 — a gain somewhere in the 0–8 %
band, unresolvable under that night's noise, with GPU occupancy unchanged (72 % vs 73 %). The
bench's isolated-block +14 % does not materialize in production: the fused elementwise glue is a
small slice of a step that also contains sequence assembly, output heads and the scheduler.

Verdict: keep the flag available, default off; re-measure the pair on a strictly idle machine
before ever claiming a number. Not a lever worth pursuing further at current sizes.

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

## Sigma-step study (2026-08-11, same seed, 576×384/124f)

Three runs, seed 0, identical prompt, 30 / 20 / 15 grid points. Changing the grid changes every
σ value, so the trajectories diverge from step 1 (PSNR 30-vs-20 ≈ 17 dB — divergence, not
quality, exactly like the quantization comparison). The judgment is therefore "equally clean?",
frame by frame:

- **Video: no visible degradation at any of the three.** The fox, fur detail and background
  stay crisp at 15 points. Wall times under identical conditions: 48 / 32 / 25 min.
- **Audio: a flag at 15.** Peak level holds from 30 to 20 points (−35.0 → −34.6 dB) then drops
  10 dB at 15 (−45.3 dB). Could be legitimate trajectory divergence (a quieter scene), but the
  30→20 stability against the 20→15 drop makes fewer-steps audio degradation the more likely
  reading. The audio scheduler's shift-3 grid loses proportionally more of its low-σ tail than
  the video's shift-12 grid does.

- **Motion: 15 is out.** Human review of the moving video found visible judder at 15 points —
  an artifact stills can never catch, which is why the frame grid above looked clean. Frame-level
  comparison is necessary but NOT sufficient: motion smoothness must be judged on the playing
  video. Together with the 10 dB audio drop, 15 points is rejected outright.

**Final verdict (human-reviewed on the moving videos): 20 sigma points are smooth** — video,
motion and audio all hold. 15 is rejected (judder + the audio drop). **The CLI default stays
at 50 deliberately**: it is the released pipeline's default and the baseline every comparison
in this repo is built against; changing it would silently pollute future baselines. Pass
`-s 20` explicitly for practical runs — a third off the 30-step quality runs, 60 % off the
default.

## Attention-mass audit (2026-08-12) — the sparse premise, measured

`H3_ATTN_AUDIT=3 generate …` (24 005-token request, 23 200 media rows audited, denoising step 3,
512 sampled query rows × all 56 heads, blocks of 64). Two curves per layer: mass captured by the
top-k blocks ranked by true mass (oracle) versus ranked by **MSA's parameter-free proxy** (block
max of raw scores).

| layer | 10 % of blocks | 20 % | 30 % | 50 % |
|---|---|---|---|---|
| 0 | 0.53 / 0.52 | 0.68 / 0.67 | 0.77 / 0.76 | 0.88 / 0.88 |
| 12 | 0.82 / 0.80 | 0.90 / 0.90 | 0.94 / 0.94 | 0.98 / 0.98 |
| 25 | 0.83 / 0.81 | 0.92 / 0.91 | 0.95 / 0.95 | 0.99 / 0.98 |
| 37 | 0.78 / 0.76 | 0.87 / 0.86 | 0.92 / 0.91 | 0.97 / 0.96 |
| 49 | 0.83 / 0.83 | 0.90 / 0.89 | 0.93 / 0.93 | 0.97 / 0.97 |

(each cell: oracle / proxy)

Three conclusions:

1. **The mass is concentrated where it matters.** From layer 12 on, 30 % of the blocks carry
   92-95 % of the attention mass. Layer 0 is the exception — diffuse (77 % at 30 % kept) — so a
   port should keep the first layer(s) dense, which sparse recipes do anyway.
2. **MSA's selection rule works without any learned weights**: the proxy sits within 1-2 points
   of the oracle everywhere. Nothing is missing from the checkpoint to reproduce the selection.
3. **The expected gain is real but bounded**: attention at ~50 % of block compute × keeping 30 %
   of blocks ≈ **–35 % of a step at the full canvas**, more if attention's true share at 40k
   tokens exceeds the bench's 49 % (the tokens^1.9 scaling suggests it does). Consistent with
   the ~2× ceiling, not the 6× the plan once assumed.

Caveat: captured mass is a proxy for fidelity, not a guarantee — the discarded 5-8 % could still
matter perceptually. The decision point (build or drop) is passed; the next checkpoint is a
same-seed comparison with a block-sparse prototype.

## Block-sparse attention, production A/B (2026-08-12): a net loss as implemented

Same-seed pair at 24 005 tokens (576×384×345f, -s 7, qint8): full attention **5 m 55 /step**,
block-sparse 30 % (audit-validated indexer: pooled-q × full keys, block-max, per-head top-k,
layer 0 dense, exact tail handling) **9 m 27 /step** — **1.6× slower**, peak memory unchanged
(34.4 GB both).

The isolated primitive benched 2× *faster*. The gap is gather granularity: the bench used
head-shared selection (rows of heads×128 = 14 KB per gather), production uses the per-head
selection the quality audit demands (rows of 128 values = 256 bytes) — 56× more rows, 56×
smaller, plus per-layer transpose copies the bench prepared off the clock. Without a fused
kernel, fine-grained gathers dominate.

PSNR between the two runs is 17 dB — trajectory divergence as expected (80-90 % captured mass
per layer compounds over 49 sparse layers); quality judgment is moot while speed loses.

**Paths forward, in order**: (1) measure head-GROUP selection quality in the audit (groups of
4-8 heads sharing a selection would fatten gather rows 4-8×; quality sits somewhere between
per-head and the collapsed head-shared curve — measure before building); (2) if grouping fails,
this optimization needs a fused Metal kernel — which is MSA's own answer — and moves from
"pure-MLX afternoon" to "kernel project". The implementation stays in the tree
(`--sparse-attention`, off by default) as the substrate for either.

## Head-group selection (2026-08-12) — the pure-MLX sparse path is exhausted

Last cheap card, measured: adjacent-head groups sharing one selection (fattening gather rows
4-8×, the production bottleneck). At 30 % blocks kept, groups of 4 lose 9-12 points of captured
mass on the layers where per-head selection mattered (layer 12: 86 → 74 %; layer 49: 90 → 81 %),
landing much closer to the collapsed shared-56 curve than to per-head. H3's heads genuinely
attend to different places; no sharing granularity reconciles gather efficiency with selection
fidelity. (Layer 37 is the lone exception — its heads agree, every mode ties at ~80 %.)

**Investigation closed.** Block-sparse attention on this checkpoint needs per-head selection,
and per-head selection at speed needs a fused kernel that never materializes the gather —
which is exactly what MSA's CUDA release is. The complete substrate stays in the tree
(`--sparse-attention`, indexer, audit tooling) for a future Metal-kernel project or a usable
upstream release (action plan 405 watches). Until then, the standing win is `-s 20` (−33 %).

---

## Canonical references (2026-08-15) — and what they are NOT

`minimax-h3 bench --tokens N --quant qint8`, measured on a **cold GPU** with five idle minutes
before each point (mandatory — see [gpu-burst-vs-sustained](../pitfalls/gpu-burst-vs-sustained.md)).
Stored machine-readably in [bench-reference.tsv](bench-reference.tsv); `scripts/bench-guard.sh`
diffs against it at ±20 % and refuses to compare if the GPU reads busy at the start.

| tokens | canvas | primitives total | measured step | bench vs reality |
|---|---|---|---|---|
| 3 712 | guard point | 25.0 s | — | — |
| 8 998 | 576×384 / 124f | 76.1 s | 62.9 s | **+21 %** |
| 24 005 | 576×384 / 345f | 353.4 s | 401 s | **−12 %** |
| 39 930 | 1344×768 / 124f | 970.8 s | ~1 080 s (≈18 min) | **−10 %** |

Reproducibility of the 9k point: 76.2 s and 76.1 s twenty minutes apart, against 75.5 s five days
earlier — under 1 % of drift, which is what makes a ±20 % tripwire meaningful.

**The sign flips with size, so this table is a tripwire and not a cost model.** The primitive sum
overestimates the real step at 9k and underestimates it at 24k and 40k: small canvases pay ~18 %
of glue the bench does not model, large ones pay a sustained-clock penalty the bench does not
either, and the two errors point opposite ways. Detect movement with it; project a run with the
measured step column, or better, with the tokens^1.9 law above.
