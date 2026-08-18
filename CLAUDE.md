# CLAUDE.md

## Project Overview

Swift MLX port of **MiniMax-H3** (MiniMaxAI/MiniMax-H3): joint video + synchronized stereo audio
generation on Apple Silicon. 33B guidance-distilled DiT + Qwen3-VL-32B conditioner + video VAE
(f16t4d24, ViT decoder) + audio VAE (32 kHz stereo). Targets `t2va` then `fl2va` (ref2va out of scope).

**INSTRUCTIONS.md is the porting spec** — read it before touching model code. It captures the
complete architecture, the packing geometry, and the fidelity traps extracted from the diffusers
reference (PR #14355, **merged into diffusers `main` on 2026-08-05**; not in a tagged
release yet — v0.39.0 predates it).

## Build & Test Commands

Build with `xcodebuild`, NOT `swift build` (SPM binaries can't find MLX's `default.metallib`
— same pitfall as the ltx/flux repos):

```bash
xcodebuild -scheme minimax-h3 -configuration Release -derivedDataPath .xcodebuild build
# binary: .xcodebuild/Build/Products/Release/minimax-h3
```

`swift build` is fine for compile checks only.

## Tests

```bash
# Everything that needs no checkpoint (geometry, mrope layout, presentation, patchify): ~0.4 s
xcodebuild test -scheme minimax-h3 -destination 'platform=macOS' -derivedDataPath .xcodebuild

# Plus the checkpoint-backed smoke tier. `xcodebuild test` does NOT forward the environment,
# so run the bundle directly or the suite silently skips:
xcodebuild build-for-testing -scheme minimax-h3 -destination 'platform=macOS' -derivedDataPath .xcodebuild
H3_MODELS_DIR=... xcrun xctest .xcodebuild/Build/Products/Debug/MiniMaxH3Tests.xctest
```

`swift test` runs the pure-Swift tests then dies on the metallib as soon as MLX is touched.
Keep the two tiers separate: anything needing weights goes in `SmokeTests.swift` behind the
`H3_MODELS_DIR` gate, so a contributor without the 163 GB download still gets a green suite.

## Model weights

`$H3_MODELS_DIR` (diffusers layout, e.g. an external SSD): `transformer/` (61.7 GB bf16),
`text_encoder/` (66.5 GB, Qwen3-VL-32B), `vae/` (10.4 GB), `audio_vae/` (0.6 GB), `tokenizer/`,
`processor/`, `scheduler/`, `audio_scheduler/`. Reference Python sources for the VAEs are in the
HF repo under `FL2VA/video_vae/*.py` and downloaded alongside.

**This repo is public: never commit a local absolute path.** Write `$H3_MODELS_DIR/...` for
checkpoint files and repo-relative paths for everything else — no volume names, no home
directories, in code, docs or knowledge notes. Detached run scripts and their logs live in
`.local-runs/` (gitignored) precisely so they can carry machine-specific paths.

## Hardware constraints (M3 Max, 96 GB)

Full-precision bf16 works ONLY with strict sequential stage loading: text encoder (~52 GB truncated
to 50 layers) → unload → transformer (61.7 GB) → unload → VAEs. Never two large components resident.
`Memory.clearCache()` between stages. See INSTRUCTIONS.md hardware envelope.

## Architecture

```
Sources/MiniMaxH3/
  MiniMaxH3.swift            # umbrella: version, H3Error, H3Debug
  Pipeline/Packing.swift     # packed-sequence geometry (H3Constants/H3Geometry/H3Packing) — THE model contract
  Scheduler/H3Scheduler.swift
  Models/Transformer/        # H3Transformer (50 blocks + 2 refiner, AdaLN table, MM-RoPE)
  Models/TextEncoder/        # Qwen3VLTextEncoder (layers 0..<50, unnormalized output)
  Models/VAE/                # video VAE: causal CNN encoder (keyframes) + 36-block ViT decoder
  Models/AudioVAE/           # audio VAE (BigVGAN-style, 32 kHz stereo)
  Loading/WeightLoader.swift # sharded safetensors + key remap + strict diagnostics
Sources/MiniMaxH3CLI/        # generate, enhance, mux, parity, bench, bench-decode, smoke, info
```

## Key implementation facts (do not re-derive)

- Text conditioning = **unnormalized** hidden state after Qwen3-VL layer 50; t2va prompt is
  encoded **verbatim** (no chat template, no special tokens).
- One packed sequence `[text | keyframe cond | audio | video]`, full attention, no CFG.
- Two schedulers: video shift 12.0, audio shift 3.0; velocity is data-ward (`x0 = x_t + σ·v`).
- Frames `17n+5` @24fps (5–15 s), latents `5n+2`; audio 40 latents/s ×2 channels channel-major.
- Rotary clock shared between video (5/3 per frame, non-uniform (1,4,4,4,4)·5/3 per latent) and
  audio (1 per latent) — this IS the AV sync. Grids built in Double.
- Mixed-precision checkpoint: `proj_in/audio_proj_in/time_embedder/proj_out/audio_proj_out` fp32,
  rest bf16. AdaLN: SiLU in fp32 → cast bf16 → linear. `norm_out` order: shift THEN scale.
- Weight key remaps: `to_out.0→to_out`, `ff.net.0.proj→ff.proj`, `ff.net.2→ff.out`.
- All checkpoint RMSNorms are affine (have weights).

## Performance work — read this before measuring anything

**The GPU's clock state moves results by up to 10×.** The same Release binary, same shapes, same
Mac on AC power, measured 1.14 / 11.1 / 2.25 / 1.70 s per identical unit of work depending on when
it ran — and came back to 1.139 after five idle minutes. A short bench on a cold GPU therefore
*overstates* the machine for any sustained phase. Full account:
`docs/knowledge/pitfalls/gpu-burst-vs-sustained.md`. Practical rules:

- Cool down before **every** point, not just the first, and alternate variants (A/B/B/A).
- Always carry a control sample in the same thermal state. That discipline has reversed three
  conclusions in this repo; without it, the second variant loses on thermals, not on merit.
- Judge a difference only against the intra-variant spread (~±2 % with the protocol).

Tools, none of which need the 163 GB checkout:

```bash
minimax-h3 bench --tokens 8998 --quant qint8   # denoising-step primitives, ~19 s
minimax-h3 bench-decode --batches 1,2,3,6      # video VAE decode passes (+ --real-vae for the checkpoint)
scripts/bench-guard.sh 8998 20                 # regression tripwire vs docs/knowledge/benchmarks/bench-reference.tsv
```

`bench-guard.sh` self-calibrates: it replays the cheap 3712-token point until the machine is back
in its reference state, and refuses to conclude otherwise. Update `bench-reference.tsv` only with
a fresh cold measurement, and say why in `docs/knowledge/log.md`.

Two closed investigations to avoid re-opening blind: block-sparse attention (per-head selection is
required for fidelity and makes the pure-MLX gather slower than dense — a fused Metal kernel is the
only path, issue #2) and the VAE decode (it saturates the GPU; the profiler's "49 %" is a sampled
device average, and direct sampling reads 99–100 %).

## Engineering Knowledge Base

`docs/knowledge/` is an OKF bundle (same convention as ltx-video-swift-mlx): when you root-cause
something new or measure something durable, add a concept file + a line in `docs/knowledge/log.md`.

## Reference code

Diffusers reference clone lives in the session scratchpad during active work; re-clone with:
`git clone --depth 1 https://github.com/huggingface/diffusers.git` (H3 is on `main` since
2026-08-05; the old `minimax-h3` PR branch is obsolete)
Files: `models/transformers/transformer_minimax_h3.py`, `schedulers/scheduling_minimax_h3.py`,
`modular_pipelines/minimax_h3/*.py`, `models/autoencoders/autoencoder_kl_minimax_h3{,_audio}.py`.
