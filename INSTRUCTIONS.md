# MiniMax-H3 Swift MLX — Porting Specification

Port of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) to Swift + MLX for Apple Silicon.
Reference implementation: diffusers branch `minimax-h3` (PR #14355), cloned during study; the authoritative
Python files are listed per component below.

## What the model is

Omni-modal generative system: **video + synchronized stereo audio, jointly denoised** in a single packed
sequence by one 33B dense transformer. 24 fps, 5–15 s, short edge 768 px, audio 32 kHz stereo.
Two checkpoint partitions: `transformer/` serves `t2va` + `fl2va`, `transformer_ref/` serves `ref2va`
(ported 2026-08-20; its `config.json` is byte-identical to the main one, so the loader, the quantization
filter and the prequantized export all apply unchanged — only the directory name differs). All other
components are shared.

**Milestone 1 (this port's first goal): `t2va` full precision bf16**, reproducing
`scripts/readme/reproducible-768p-t2va-request.sh` qualitatively. Then quantization (on-the-fly + prequantized
export), then `fl2va` (needs Qwen3-VL vision tower + video VAE encoder), then `ref2va` (needs the audio VAE
*encoder*, Qwen3-VL's video processor and the per-reference rotary layout — see
`docs/knowledge/playbooks/ref2va-port-plan.md`).

## Hardware envelope (M3 Max, 96 GB)

Full-precision bf16 requires *strict sequential* stage loading — never two big components resident at once:

| Stage | Component | Weights | Peak est. |
|---|---|---|---|
| 1. Text encode | Qwen3-VL-32B text model, layers 0–50 only | ~52 GB (66.5 full; last 14 layers + LM head + vision tower droppable for t2va) | ~55 GB |
| 2. Denoise | H3 transformer 33B | 61.7 GB bf16 | ~70 GB |
| 3. Video decode | Video VAE (ViT decoder) | 10.4 GB | ~15–20 GB (tiled) |
| 4. Audio decode | Audio VAE | 0.6 GB | small |

Unload + `Memory.clearCache()` between stages (LTX pattern). The macOS GPU wired limit may need raising
(`sudo sysctl iogpu.wired_limit_mb=...`) for stage 2; measure first.

Weights live in `$H3_MODELS_DIR` (diffusers layout, top-level subfolders:
`transformer/ text_encoder/ vae/ audio_vae/ scheduler/ audio_scheduler/ tokenizer/ processor/`).

## Component specs

### 1. Text encoder — Qwen3-VL-32B (`text_encoder/`)

Reference: `transformers` Qwen3-VL; config in `text_encoder/config.json`.
- Text model: 64 layers, hidden 5120, 64 heads / 8 KV heads (GQA), head_dim 128, intermediate 25600 (SwiGLU),
  RMSNorm eps 1e-6, rope_theta 5e6, **mrope interleaved** with sections [24, 20, 20], vocab 151936,
  untied LM head (unused).
- **Conditioning = `hidden_states[50]`**: the *unnormalized* residual stream after decoder layer 50
  (index 50 with 0 = embeddings). NOT the final norm output. → layers 51–63, final norm, LM head are dead
  weight for our use; skip loading them.
- t2va presentation: **verbatim prompt tokens, `add_special_tokens=False`, no chat template**. All rows
  tagged TEXT (1).
- fl2va presentation (milestone 3): per keyframe, prepend `"<Picture i>: "` label (TEXT tag) + vision block
  `<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>` (VIDEO tag 0), then the prompt. Vision tower:
  27-layer ViT, hidden 1152, patch 16, spatial_merge 2, deepstack indexes [8,16,24], out 5120. mrope token
  type ids: 0 text, 1 image.
- Pure text with interleaved-mrope: positions are effectively 1-D (t=h=w), verify against transformers'
  `Qwen3VLTextModel` position handling.
- Tokenizer: Qwen2 BPE from repo's `tokenizer/` (has H3's added special tokens like `<d>`); use
  swift-transformers `Tokenizers` pointed at the local files.
- Output dtype: embeddings emitted in the encoder's dtype (bf16), shape `(1, L, 5120)`.

### 2. Transformer — `MiniMaxH3Transformer3DModel` (`transformer/`)

Reference: `src/diffusers/models/transformers/transformer_minimax_h3.py` (644 lines, self-contained).
Config `transformer/config.json`: 50 layers + 2 refiner layers, hidden 5376, 56 heads × 128 (inner dim
7168 > hidden), ffn 14336 SwiGLU (bias-free), in_channels 24 video / 32 audio, patch (1,2,2), text_dim 5120,
freq_dim 256, time_embed_hidden 5376 → time_embed_dim 2688, rope_freq_dim 16, rope_theta 10000, all eps 1e-5.

Structure per block: `norm1` (RMSNorm, no affine? — check keys) → AdaLN scale/shift → self-attn
(to_q/to_k/to_v/to_out bias-free, per-head RMSNorm on q and k over head_dim) → gated residual →
`norm2` → AdaLN → SwiGLU FF → gated residual. `adaln_proj.linear`: 2688 → 6·5376·3 (six params ×
3 modalities), rows laid out `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, addressed by
`timestep_index * 3 + token_tag`.

- **MM-RoPE**: one shared `inv_freq` of 16 freqs (`theta^(2i/32)`), each of (t,h,w) contributes 16 angles,
  concat → 48, duplicated → 96 of the 128 head channels rotated, rotate_half convention (non-interleaved),
  remaining 32 channels pass through. Positions cast to float32 before multiply.
- Modality-specific pieces only: `proj_in` (96→5376, video patches), `audio_proj_in` (32→5376),
  `context_embedder` (5120→5376) + 2-block **token refiner** (plain pre-norm blocks, no AdaLN/RoPE, final
  RMSNorm) on the text stream, `norm_out` (RMSNorm + linear 2688→2·5376, **shift-then-scale order**, indexed
  per row by timestep_indices), `proj_out` (5376→96), `audio_proj_out` (5376→32). Both heads run over the
  full sequence, rows selected after.
- **Mixed-precision checkpoint**: `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out`
  are float32; everything else bf16. Keep them float32; the packed stream runs at the text embedder's dtype
  (bf16). Timestep MLP: `time_proj` sinusoidal (256, flip_sin_to_cos=true, downscale_freq_shift=0, input
  t∈[0,1] **unscaled**) → linear 256→5376 → SiLU → linear 5376→2688; runs float32; each AdaLN applies
  SiLU(temb) in fp32 then casts to bf16 before its projection.
- Forward inputs: packed rows per modality + `timestep` (distinct values), `timestep_indices` (seq_len),
  `token_tags` (seq_len; 0 video, 1 text, 2 audio), `position_ids` (seq_len, 3), and the three index tensors.
  Sequence assembled by scatter (index_copy) at the packed positions. **We build padless sequences → no
  attention mask, plain SDPA full attention.**
- Output: video velocity rows + audio velocity rows (data-ward sign).

### 3. Scheduler — `MiniMaxH3Scheduler` (×2 instances)

Reference: `scheduling_minimax_h3.py` (288 lines). Rectified-flow Euler, eta=0.
- Sigma grid: `linspace(1, 0, num_inference_steps)` → shift `σ' = s·σ / (1 + (s−1)·σ)` →
  collapse consecutive float32 duplicates. `timesteps = 1 − sigmas[:-1]` → N−1 model evals for N grid points.
- Video shift **12.0**, audio shift **3.0** (both configurable per request).
- Step (float32): `x0 = x_t + (1−t)·v` (**plus** sign, data-ward velocity), then
  `x_next = r·x_t + (1−r)·x0`, `r = σ_next/σ` (σ from the grid, not recomputed from t).
- `scale_noise(x0, t, noise) = t·x0 + (1−t)·noise` (t=1 clean) — used for keyframe noise-aug.

### 4. Packing (`packing.py`, 538 lines — port faithfully, it IS the model geometry)

Row order: `[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]`.
- Canvas: short edge 768, area cap 768·1344, round both axes to ×32, aspect 1:4…4:1. Default 16:9 (t2va).
- Frames: snap up to `17n+5` (default request 124 → snaps to 124? 124 = 17·7+5 ✓). Latent frames `5n+2`;
  audio latents `round(frames/24·40)` per channel, ×2 channels (channel-major rows: all L then all R).
- Position grid **built in float64** (Double in Swift), cast f32 only inside RoPE:
  - Text rows: t = row index (0…L−1), h = w = 0.
  - Media clock starts at t = L. Video latent frame k spans non-uniform steps `5/3 · (1,4,4,4,4)` cyclic
    (mirrors 17→5 VAE grouping); audio rows advance 1.0 per latent (40 Hz ↔ 24 fps·5/3 — shared clock =
    AV sync). Audio h = 0; audio w pinned to `width_grid[0]` (L channel) / `width_grid[-1]` (R channel).
  - Spatial grids: `linspace(left, left+ratio, dim/patch, endpoint=false)·32` with `ratio = dim/√(h·w)`,
    `left = (1−ratio)/2` — reproduce numpy `endpoint=False` semantics exactly: `start + i·(stop−start)/num`.
  - Keyframe condition block: t anchored at first frame time (`"first"` → L) or last
    (`L + span(F) − 5/3`), same spatial grid as a target frame.
- `build_row_timesteps`: video rows at video t, audio rows at audio t, text rows inherit video t, condition
  video rows pinned at `max(video_t, 0.999)`; reduce to (sorted unique values, per-row indices).
- Patchify video latents `(1,C,F,H,W)` → rows `(F/pt·H/ph·W/pw, C·pt·ph·pw)` frame-major then row-major
  (permute order (0,2,4,6,1,3,5,7)); exact inverse for unpatchify. Audio unpack: rows
  `(2·A, 32)` → `(2, 32, A)`.
- Constants: keyframe noise-aug t = 0.999 (constant every step); keyframe VAE posterior sampled with
  fixed seed 42 (independent of request seed); sampled latent **rounded through float16** before
  normalization (part of the released conditioning contract).

### 5. Video VAE — `AutoencoderKLMiniMaxH3` (`vae/`)

Reference: `autoencoder_kl_minimax_h3.py` (922 lines) + original sources in `FL2VA/video_vae/*.py`.
f16 spatial / t4 temporal / 24 latent channels, temporally causal, `clip_length` 17 (encodes 17-frame
chunks → 5 latents, drops 3 trailing per chunk: `token_drop`), CNN encoder (block_out
128,256,256,512,512,1024) + **36-layer ViT decoder** (heads 32×64, ffn ×4, 4 register tokens,
rope_theta 100, rope_dim_ratio 0.75) ≈ 5B params.
- Latents normalized per-channel with `latents_mean`/`latents_std` from config (24 values each).
- Pixel convention: ImageNet-normalized RGB over [0,1] (mean .485/.456/.406, std .229/.224/.225) — revert
  after decode, clamp to [0,1].
- Reference decodes under fp16 autocast (weights fp32) — we can run bf16/fp16 compute.
- Decode temporally in chunks (causal) to bound memory; spatial tiling exists in reference for the encoder.
- Milestone 1 needs the **decoder only**.

### 6. Audio VAE — `AutoencoderKLMiniMaxH3Audio` (`audio_vae/`)

Reference: `autoencoder_kl_minimax_h3_audio.py` (679 lines) + `FL2VA/audio_vae/*.py` (DAC/BigVGAN family).
Mono model; stereo = 2 batch items, recombined after. 32 kHz, hop 800 → 40 latents/s, 32 latent channels.
Has its own `latents_mean`/`latents_std` in config. Milestone 1 needs the **decoder only**.
Output `(2, samples)` stereo waveform → mux with video via AVFoundation (AAC track in the MP4).

### 7. Pipeline flow (t2va)

1. Resolve canvas (16:9 → 1344×768? verify: ratio>1 → width=768·16/9=1365.33, area cap → scale, round →
   1344×768) and frames (default 124 → latent F=37, audio A=207 (124/24·40 rounded), rows: V=37·(48·42÷4)…
   compute exactly in code).
2. Tokenize prompt verbatim → Qwen3-VL layers 0–50 → embeds (1, L, 5120). **Unload encoder.**
3. Build packed layout + position grid + row timestep plan (per step: unique values + indices).
4. Draw noise: video latent tensor `(1,24,F,h/16,w/16)` then audio rows `(2A, 32)` (order matters for seed
   reproducibility; MLX RNG ≠ torch RNG — cross-check via exported latents when validating).
5. Load transformer. Per step (one forward, no CFG): scatter rows → 52 blocks → heads → velocities;
   scheduler.step on generated video rows (video sched) and audio rows (audio sched). Conditioning rows
   never updated. **Unload transformer.**
6. Video: unpatchify → denorm (mean/std) → VAE decode (chunked) → un-ImageNet → frames.
7. Audio: unpack channel-major → denorm → audio VAE decode per channel → stereo wav.
8. Mux MP4 (24 fps + 32 kHz stereo AAC) via AVFoundation.

## Fidelity traps (collected during study — do not lose)

- Text conditioning is **pre-norm** hidden state at layer 50; a 50-layer truncated stack's "last hidden
  state" would be post-norm → wrong.
- AdaLN activation precision: SiLU applied to fp32 temb, THEN cast bf16 (biases every block identically
  otherwise).
- norm_out order is **shift, scale** (not scale, shift).
- Scheduler σ for `x0` comes from `1 − t` at the *sample's* dtype; the Euler ratio uses the fp32 grid —
  keep both sources as reference does.
- Position grids in float64; numpy-`linspace(endpoint=False)` semantics; pairwise-sum for the "last"
  keyframe anchor span (ulp-level, matters only for bit-exactness).
- Keyframe latents: sampled posterior (seed 42) → **cast through fp16** → normalize.
- Timestep values are t = 1−σ in [0,1], NOT scaled by 1000.
- `num_inference_steps` counts grid points → N−1 forwards.
- Audio rows channel-major `[all L | all R]`; audio VAE is mono, batch of 2.
- Modules to keep unquantized (from upstream int8 recipe): transformer `proj_in, audio_proj_in,
  context_embedder, time_embedder, time_proj, token_refiner, norm_out, proj_out, audio_proj_out`;
  text encoder `visual, embed_tokens, final norm, lm_head`.

## Quantization roadmap (after full-precision validation)

Follow flux-2-swift-mlx patterns: `TransformerQuantization` enum (qint8/int4/mxfp8/nvfp4…) with
`descriptor` switch; on-the-fly `quantize(model:groupSize:bits:mode:)` after weight load; prequantized
export à la `Flux2PrequantizedCheckpoint` (`<modelDir>/mlx-prequantized/<quant>/<component>.safetensors`
with metadata + validation + atomic replace). Text encoder benefits the most per GB (only run once).
The AdaLN projections (~13B params) are quantizable OR precomputable per (step × block) — evaluate both.

## Validation assets

- `scripts/readme/reproducible-768p-t2va-request.sh` (prompt/params) vs `assets/t2va.mp4` — qualitative.
- For numeric parity: dump per-stage tensors from the Python reference (prompt embeds, packed layout,
  per-step velocities with fixed pre-drawn latents fed via `latents=`/`audio_latents=`) and compare in
  Swift tests. Feeding identical noise sidesteps the torch-vs-MLX RNG mismatch.
