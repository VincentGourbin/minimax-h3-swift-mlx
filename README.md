# minimax-h3-swift-mlx

> ⚠️ **Work in progress** — the t2va pipeline works end-to-end and is numerically validated
> against the reference implementation, but APIs, CLI flags and performance are still moving.

Swift MLX port of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) for Apple Silicon:
**joint video + synchronized stereo audio generation** — one 33B guidance-distilled transformer
denoises a single packed sequence holding text, audio and video rows at once. 24 fps, 5–15 s,
768 px short edge, 32 kHz stereo, no CFG, no vocoder.

## Examples

| 🦊 [Fox in snow](docs/examples/t2va/fox-snow-960x544.mp4) | 🥁 [Drum solo](docs/examples/t2va/drum-solo-768x768.mp4) |
|---|---|
| ![fox](docs/examples/t2va/fox-frame.png) | ![drums](docs/examples/t2va/drums-frame.png) |

Details and prompts in [docs/examples/t2va](docs/examples/t2va/README.md).

## Status (August 2026)

**Working:**
- Full t2va pipeline: tokenizer → Qwen3-VL-32B conditioner (layers 0–49 only) → 33B
  Omni-Transformer → ViT video VAE (tiled + temporally chunked, fp16) → BigVGAN audio VAE →
  MP4 mux (H.264 + AAC) via AVFoundation
- Strict sequential loading fits full-precision bf16 in 96 GB unified memory
- **Numeric parity vs the diffusers reference** (`minimax-h3 parity`, dumps via
  `scripts/parity_reference.py`): audio VAE max|Δ| 5e-6, video VAE 1.3e-5 (fp32) / 1.1e-2 (fp16
  autocast recipe), packing grid exact (Δ=0), DiT block-0 video & audio velocities at bf16 noise
- Built-in profiling (`--profile`, [swift-mlx-profiler](https://github.com/VincentGourbin/swift-mlx-profiler)):
  console report + Perfetto trace
- Crash-safe generation: raw frames+audio saved as `.raw.safetensors` before muxing;
  `minimax-h3 mux` rebuilds an MP4 anytime (`--normalize-audio` for quiet scenes)

**Timings (M3 Max, bf16, full precision):**
- ~23 s/step at 960×544×22 frames (~3.7k packed tokens); ~4.5 min/step at 960×544×124 frames
  (~17.4k tokens — attention is ~75 % of a step at that length)
- Weight loads: ~2 min 20 total from external USB SSD (profiled; quantization will cut this)

**Quantization** (`--transformer-quant` / `--text-encoder-quant`, qint8/qint6/int4, MiniMax's
official module exclusions; `export-quantized` writes reusable prequantized checkpoints that
loaders pick up automatically):
- Prequantized qint8 text encoder: **load 55.7 s → 2.6 s**, file 52 → 27.5 GB, output identical
- Quantized full pipeline: peak memory 63 → 34 GB, but MLX quantized matmuls are markedly slower
  than bf16 GEMMs at DiT sequence lengths — on 96 GB the recommended config is prequantized
  qint8 text encoder + full-precision transformer

**Not yet:**
- fl2va (first/last keyframe): needs the Qwen3-VL vision tower + the causal video VAE encoder
- ref2va, and the hosted-only H3-Context-IR / H3-Regenerate-2K stages
- MiniMax's sparse attention (not yet open-sourced) — the real unlock for full 768p speed

## Requirements

- Apple Silicon Mac, **96 GB+ unified memory** for full-precision bf16 (quantized paths later)
- macOS 15+, Xcode 16+
- ~145 GB of weights (diffusers layout)

## Getting the weights

```bash
export H3_MODELS_DIR=/path/to/models/MiniMax-H3   # or pass --models-dir to every command
hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" --include "scheduler/*" --include "audio_scheduler/*" \
  --include "tokenizer/*" --include "processor/*" --include "text_encoder/*" \
  --include "transformer/*" --include "vae/*" --include "audio_vae/*" \
  --local-dir "$H3_MODELS_DIR"
```

## Build

Build with `xcodebuild` (SPM binaries can't locate MLX's metallib):

```bash
xcodebuild -scheme minimax-h3 -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .xcodebuild build
alias minimax-h3=$PWD/.xcodebuild/Build/Products/Release/minimax-h3
```

## Usage

```bash
# Sanity: load every component against the real checkpoint + tiny forwards
minimax-h3 smoke all

# Text -> video+audio (768p 16:9 canvas, 124 frames ≈ 5.2 s, 50 sigma points)
minimax-h3 generate "A red fox trotting through a snowy pine forest, snow crunching underfoot" \
  -o fox.mp4 --normalize-audio

# Smaller canvas = much faster per step (multiples of 32). Short edge 768 is the trained
# regime; smaller trades quality (and ~6 dB of audio level) for speed.
minimax-h3 generate "..." -W 960 -H 544 -s 30 -o fast.mp4 --profile

# One-time: export a prequantized text encoder, then every run loads it in seconds
minimax-h3 export-quantized text-encoder --quant qint8
minimax-h3 generate "..." --text-encoder-quant qint8 -o out.mp4

# Rebuild an MP4 from a saved raw result (written automatically before every mux)
minimax-h3 mux fox.raw.safetensors -o fox2.mp4 --normalize-audio
```

**Prompting**: H3 is trained on structured Context-IR prompts (`integrated_multimodal_description:`
/ `overall_soundscape:` / `non_diegetic_music:` sections, `[Shot N]` timecodes). Audio loudness is
scene-faithful — quiet scenes decode quiet. See
[docs/knowledge/playbooks/context-ir-prompting.md](docs/knowledge/playbooks/context-ir-prompting.md)
and MiniMax's prompt guides (`docs/` in the HF repo).

## Architecture

| Component | What | Size (fp) |
|---|---|---|
| Conditioner | Qwen3-VL-32B, **unnormalized hidden state after layer 50** (layers 51+ never loaded) | ~52 GB bf16 |
| Omni-Transformer | 33B dense single-stream DiT, per-(timestep, modality) AdaLN, 3-axis MM-RoPE | 61.7 GB bf16 |
| Video VAE | f16×t4×d24; causal CNN encoder (not yet ported), 36-layer ViT decoder, tiled+chunked | 10.4 GB (fp16 decode) |
| Audio VAE | DAC encoder (not yet ported) / BigVGAN decoder, mono ×2 for stereo, 40 latents/s | 0.6 GB fp32 |

The packed sequence `[text | keyframe cond | audio | video]` shares one rotary clock between
audio (1 unit/latent @40 Hz) and video (5/3 units/frame @24 fps) — that clock *is* the AV sync.
`INSTRUCTIONS.md` holds the full porting spec and fidelity notes; `docs/knowledge/` collects
measured pitfalls and playbooks.

## Sibling projects

[flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) ·
[ltx-video-swift-mlx](https://github.com/VincentGourbin/ltx-video-swift-mlx) ·
[swift-mlx-profiler](https://github.com/VincentGourbin/swift-mlx-profiler)

## License

Code: MIT. Model weights are covered by the
[MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
