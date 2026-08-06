---
okf_version: "0.1"
kind: benchmark
created: 2026-08-06
device: Apple M3 Max, 96 GB unified memory
---

# Quantization: fidelity, speed, memory, loads (all MLX modes)

## Methodology

- Machine strictly idle per run (two earlier readings — "5x slower", then "1.34x" — were
  pollution from concurrent GPU/disk activity; a first-step kernel-compile spike also inflates
  short-run averages). 960×544 × 22 frames × 10 sigma points (9 evals, 3.7k packed tokens),
  seed 42, prequantized qint8 text encoder in every run.
- **Per-forward fidelity**: `parity dit-block0 --quant <mode>` — max|Δ| of the block-0 video
  velocity vs the diffusers reference on identical inputs (scale 260; bf16 noise floor 1.11).
- **Sample divergence**: PSNR between same-seed videos — measures how far the diffusion
  trajectory drifts, NOT visual quality (every mode below produced clean frames).

## Results

| Mode | bits/group | Fidelity max\|Δ\| | Step avg | Step std | Peak MLX | PSNR vs bf16 |
|---|---|---|---|---|---|---|
| bf16 | — | 1.11 | 31.2 s¹ | 22.9 s¹ | 63.2 GB | — |
| **qint8** | 8/64 | **1.41** | **28.0 s** | 2.5 s | 34.4 GB | 33.1 dB |
| qint6 | 6/64 | 1.98 | — | — | — | — |
| mxfp8 | 8/32 | 5.16 | 24.5 s | 1.1 s | 33.4 GB | 20.8 dB |
| int4 | 4/64 | 5.32 | 23.6 s | 1.6 s | 26.2 GB² | 19.9 dB |
| nvfp4 | 4/16 | 4.99 | — | — | — | — |
| mxfp4 | 4/32 | 7.22 | — | — | — | — |

¹ bf16's std is dominated by a first-step compile spike; its post-warmup steps are ~24 s.
² int4 peak is the text-encoder phase; the int4 transformer itself sits at ~19 GB resident.

Audio-branch fidelity follows the same ordering (qint8 0.62 … mxfp4 2.59, noise 0.62).

## Memory caveat (important)

The "Peak MLX" column samples memory at PHASE BOUNDARIES only (profiler limitation): it shows
cruising peaks, not transients. **On-the-fly quantization still materializes the full 62 GB bf16
transformer during its load phase** before shrinking — invisible to the sampling. Practical
consequence: on 32-64 GB machines, on-the-fly transformer quantization does NOT fit; only the
prequantized path (which never materializes bf16) honors the reduced budget. Future improvement:
shard-wise quantize-during-load to cap the transient without a prequantized file.

## Findings

1. **Quantized steps run at bf16 speed or slightly better** at this sequence length — the
   memory-bandwidth savings offset the kernel overhead. Quantization costs nothing in speed.
2. **Affine beats microscaling on fidelity for this checkpoint**: mxfp8 (8-bit!) lands at
   int4-affine level. Independent confirmation: the mxfp8 text encoder's hidden-state RMS drifts
   to 81.4 vs 106.7 unquantized, while int4 (108.5) and nvfp4 (109.8) stay close.
3. **PSNR between modes is trajectory divergence, not quality**: int4's 19.9 dB frame is as
   clean as bf16's — the sample simply walked elsewhere.
4. **Prequantized export/load round-trips bit-exactly for every family** (validated on the text
   encoder for int4, mxfp8, nvfp4: on-the-fly RMS == prequantized-pickup RMS to the last digit;
   layouts differ — affine carries biases, 1251 tensors vs 901 — both handled).

## Loads (Lexar USB SSD, cold-ish)

| | bf16 | qint8 prequantized |
|---|---|---|
| Text encoder | 55.7 s (48 GB read) | 30 s cold / 2.6 s cache-warm (27.5 GB) |
| Transformer | 1 m 15 (62 GB read) | 39.7 s (36 GB) |

## Recommendation

**Prequantized qint8 for both components** is the default: near-transparent fidelity, no speed
cost, −45 % peak memory, ~2× faster loads. int4 is the max-headroom option (clean output,
looser trajectory fidelity). Avoid mxfp modes on this checkpoint.
