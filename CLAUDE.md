# CLAUDE.md

## Project Overview

Swift MLX port of **MiniMax-H3** (MiniMaxAI/MiniMax-H3): joint video + synchronized stereo audio
generation on Apple Silicon. 33B guidance-distilled DiT + Qwen3-VL-32B conditioner + video VAE
(f16t4d24, ViT decoder) + audio VAE (32 kHz stereo). Targets `t2va` then `fl2va` (ref2va out of scope).

**INSTRUCTIONS.md is the porting spec** — read it before touching model code. It captures the
complete architecture, the packing geometry, and the fidelity traps extracted from the diffusers
reference (branch `minimax-h3`, PR #14355).

## Build & Test Commands

Build with `xcodebuild`, NOT `swift build` (SPM binaries can't find MLX's `default.metallib`
— same pitfall as the ltx/flux repos):

```bash
xcodebuild -scheme minimax-h3 -configuration Release -derivedDataPath .xcodebuild build
# binary: .xcodebuild/Build/Products/Release/minimax-h3
```

`swift build` is fine for compile checks only.

## Model weights

`$H3_MODELS_DIR` (diffusers layout, e.g. an external SSD): `transformer/` (61.7 GB bf16),
`text_encoder/` (66.5 GB, Qwen3-VL-32B), `vae/` (10.4 GB), `audio_vae/` (0.6 GB), `tokenizer/`,
`processor/`, `scheduler/`, `audio_scheduler/`. Reference Python sources for the VAEs are in the
HF repo under `FL2VA/video_vae/*.py` and downloaded alongside.

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
  Models/VAE/                # video VAE (TODO)
  Models/AudioVAE/           # audio VAE (TODO)
  Loading/WeightLoader.swift # sharded safetensors + key remap + strict diagnostics
Sources/MiniMaxH3CLI/
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

## Engineering Knowledge Base

`docs/knowledge/` is an OKF bundle (same convention as ltx-video-swift-mlx): when you root-cause
something new or measure something durable, add a concept file + a line in `docs/knowledge/log.md`.

## Reference code

Diffusers reference clone lives in the session scratchpad during active work; re-clone with:
`git clone --depth 1 --branch minimax-h3 https://github.com/huggingface/diffusers.git`
Files: `models/transformers/transformer_minimax_h3.py`, `schedulers/scheduling_minimax_h3.py`,
`modular_pipelines/minimax_h3/*.py`, `models/autoencoders/autoencoder_kl_minimax_h3{,_audio}.py`.
