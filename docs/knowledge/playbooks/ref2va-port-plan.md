---
okf_version: "0.1"
kind: playbook
created: 2026-08-20
status: ready to implement — issues #8 (Phase A), #9 (Phase B), #10 (Phase C)
---

# Porting ref2va — the handoff plan

This document carries everything needed to implement `ref2va` without re-deriving it from the
Python. The contracts below were extracted from the diffusers reference on `main`
(`src/diffusers/modular_pipelines/minimax_h3/`: `references.py`, `before_encoder.py`,
`encoders.py`, `before_denoise.py`, `denoise.py`) on 2026-08-20. When this document and the Python
disagree, the Python wins — re-read it, then fix this document.

**What ref2va is**: generation conditioned on an ordered list of references — images (subject/
style), videos (motion/camera, with their soundtrack), audio clips (voice timbre, music). It uses
its own transformer weights (`transformer_ref/`, 61.7 GB — **config byte-identical to
`transformer/`**, so `H3Transformer`, the loader, quantization and the prequantized export all
apply unchanged) and a six-section Context-IR prompt.

## Ground rules for the implementing session

- Read `CLAUDE.md` first; `INSTRUCTIONS.md` for the architecture. Build with `xcodebuild`, never
  `swift build` (metallib). Two test tiers; smoke tests run via `xcrun xctest` with
  `H3_MODELS_DIR` because `xcodebuild test` drops the environment.
- Reuse before writing: the fl2va port already built the video VAE *encoder*, PIL-exact LANCZOS
  (`H3KeyframeImage.resizedLanczos`), canvas resolution (`H3Geometry.resolveCanvasSize`),
  condition-row packing/pinning, vision-tower presentation, and `scripts/parity_reference.py`
  (durable venv: `.local-runs/parity-venv`).
- Every phase lands with tests and a parity probe BEFORE the next phase starts. That discipline
  (not talent) is what made fl2va land: when the E2E conditioner parity failed at cosine 0.67, the
  per-component probes localized it in one evening.
- Never a machine-specific absolute path in the tree; run scripts and logs live in `.local-runs/`.
- Quality judgments happen on the MOVING video, by the human. Metrics filter, they never conclude
  (four documented cases where they endorsed clips the human rejected).

## Global invariants (all phases)

1. **The reference list is ordered, and the order is the request.** It fixes the `<Picture i>` /
   `<Audio j>` / `<Video k>` labels (numbered per modality, in list order) AND the rotary-clock
   layout. Do not sort, do not group by modality.
2. **Rates travel with media.** A video carries its `fps`, audio its `sampleRate`. H3 resamples
   onto its own 24 fps / audio-VAE rate; a lost rate conditions the request at the wrong speed
   with no error anywhere.
3. **Limits** (validated with errors, not clamped): ≤9 images, ≤3 videos, ≤3 audios, ≤12 total,
   ≥1 reference, and audio references never alone.
4. **References never bind the generated geometry.** No height/width given → the canvas is H3's
   own 16:9 (768 short edge, 768·1344 area cap). `num_frames` snaps UP to `17n+5`, duration
   checked on the *aligned* count, 5–15 s.
5. The packed layout is `[text | reference blocks, in order | target audio | target video]`.

## Phase A — reference types, decoding, normalization (issue #8)

New file `Sources/MiniMaxH3/Pipeline/H3References.swift` (types + normalization) and a media
decode helper in the CLI target (AVFoundation stays out of the core library if possible).

### Types

`H3Reference` enum or protocol with three cases, mirroring the dataclasses:
- `image(H3KeyframeImage)`
- `video(frames: MLXArray /* (T,H,W,3) uint8 */, fps: Double, audio: MLXArray? /* (ch,samples) */,
  sampleRate: Int?)`
- `audio(waveform: MLXArray /* (ch,samples) */, sampleRate: Int?)`
`nil` rates mean "already at H3's own rate" (24 fps / audio-VAE rate) — leave media untouched.

### File decoding (AVFoundation, replacing PyAV)

- Video: decode ALL frames to RGB uint8, take the container's average frame rate, apply the
  display-matrix rotation snapped to the nearest quarter turn (ffmpeg semantics: undo the
  counterclockwise rotation), and read the soundtrack as float32 `(channels, samples)` at its
  native rate in a second pass. Leave a non-square pixel aspect ratio alone (the reference
  explicitly refuses to correct it).
- Audio: first audio stream, float32, native rate and channel layout (mono is upmixed later, in
  normalization — NOT at decode).

### Normalization (the exact contracts)

**Video frames** (`_normalize_video_condition`), in this order — resample FIRST, rescale SECOND
(mirrors the reference's ffmpeg decode order):
1. To uint8 THWC.
2. Onto 24 fps by dropping/duplicating whole frames: `scale = 24/fps`;
   `slots[i] = floor(i*scale + 0.5)`; frame `i` repeats `slots[i+1]-slots[i]` times, the last one
   until `floor(T*scale + 0.5)`. Skip when `fps == 24`.
3. Truncate to the generated `num_frames`.
4. Canvas from the video's OWN aspect ratio through the same rule as the target
   (`resolveCanvasSize`, short edge 768, cap 768·1344, multiple 32); rescale per-frame with the
   PIL-exact LANCZOS only when the shape differs (already-right frames pass through untouched —
   that is the parity-exact route).

**Soundtracks/audio** (`_normalize_audio_condition`): float32 → truncate at the NATIVE rate to
`num_frames/24` seconds → mono upmixed to stereo by channel repeat → ONE resample to the audio
VAE rate. Truncate-then-resample, never the reverse. Resampler: match torchaudio's
(sinc/Kaiser); the parity probe decides how close is close enough.

**Images**: accepted 1:4…4:1 (error outside). Short edge to **2048** (upscaling included, NO area
cap — deliberately different from the canvas rule): `scale = 2048/min(w,h)`,
`target = max(32, round(dim*scale/32)*32)` per axis, LANCZOS. Reuse `H3KeyframeImage`.

### Done when

- No-checkpoint tests: ordering preserved, every limit enforced, fps-resample slot formula against
  hand-computed cases (incl. 23.976→24 and 30→24), truncate-before-resample, mono upmix, image
  2048 rounding (e.g. 1920×1280 → 3072×2048; a 100×500 errors), rotation quarter-turn snap.
- Parity probe `ref-normalize` in `scripts/parity_reference.py` + `ParityCommand`: one image, one
  video (odd fps, with soundtrack), one audio vs `normalized_references`. Frames bit-exact where
  no rescale happens; rescaled frames within the LANCZOS tolerance already established for
  keyframes; audio within resampler tolerance (document the number found).

## Phase B — encoding, packing, presentation (issue #9)

### Audio VAE encoder (the genuinely new model code)

`Models/AudioVAE/` has decode only. Port the encoder from the HF repo's Python
(`audio_vae` sources downloaded next to the checkpoint) into `H3AudioVAE`, behind
`includeEncoder:` like the video VAE. Contract from the reference encoder step:
- stereo waveform `(2, samples)` → encode as TWO batch items of the mono VAE (`[:, None]`);
- take the posterior **mode** (mean), never sample — no seed involved for soundtracks;
- normalize `(latents − latents_mean)/latents_std` with the config vectors, then transpose to
  `(2, T, C)` and reshape channel-major to `(T·2, C)` rows — same channel-major convention as
  `H3Packing.unpackAudioTokens`, run in reverse.
- Parity probe `audio-vae-encode` vs the Python encoder (fp32; expect ~1e-5 like the decoder).

### Visual reference encoding

Through the EXISTING `encodeClip` + `encode_vae_condition` contract from fl2va: fresh seed-42
posterior per reference, fp16 round-trip BEFORE normalization. One tensor per reference, at its
own resolution — they never share a shape, so keep them as a list.
Video references first snap their frame count DOWN (not up):
`T' = max(1, (T − 5) div 17) · 17 + 5` — so the VAE encodes without padding.

### Condition noising (video side only)

Per condition, in packed order: one fresh noise draw (request generator), then
`scale_noise(condition, t=0.999)`, then patchify. Reference soundtracks are NEVER noised — audio
conditions at t=0. The denoising loop writes only the generated rows (the fl2va pinning machinery
generalizes; pinned t stays max(σ,0.999) on video condition rows).

### Packed layout and the rotary clock (`build_ref2va_packed_sequence`)

Sequence: `[text | per-reference blocks | target audio | target video]`. Per reference block:
- image → its video rows only;
- audio → its audio rows only;
- video → its soundtrack's audio rows IMMEDIATELY BEFORE its video rows, sharing the same rotary
  origin (that alignment IS the AV sync of the reference).
Rotary clock (float64, as everywhere): starts at `num_text_tokens`; each block advances it:
- image: **+1.0** exactly (one integer slot, NOT a latent frame's 5/3);
- audio: `+latentCount` (1.0 per audio latent);
- video: `+max(audioLatents, videoSpan)` where `videoSpan` is the SEQUENTIAL float64 sum of
  `(5/3)·(1,4,4,4,4)` per latent frame. Caution: the target-video span elsewhere in the port uses
  a pairwise-sum reproduction; the two differ in the last ulp from 16 latent frames on, and the
  reference keeps BOTH, one per call site. Do the same; do not unify.
Target audio and target video then share the origin the blocks left behind.
Tags: text rows 1 except vision-block rows tagged 0 (existing dual-tagging); all audio rows
(reference and target) `audioTag`; all video rows (reference and target) `videoTag`.

### Presentation (`Ref2VATextEncoderStep`)

Before the verbatim prompt, one label per reference, numbered per modality in list order:
- `"<Picture i>: "` + one vision block (the 2048-edge image through the vision tower);
- `"<Audio j>: "` alone — audio NEVER reaches the conditioner;
- `"<Video k>: "` + one TIMESTAMPED vision block per merged frame pair: sample the normalized
  24 fps frames at 2 fps (every 12th), pad the tail to the temporal patch (2) by repeating the
  last sampled frame, merge in pairs; each block is prefixed `"<{t:.1f} seconds>"` where t is the
  MEAN of the pair's timestamps formatted with round-half-to-even (first pair of a 2 fps read =
  `<0.2 seconds>`, not 0.3). Minimum length check: at least `temporal_patch` sampled frames.
The six-section prompt format is already documented in
[context-ir-prompting](context-ir-prompting.md); the presentation does not parse it — it goes in
verbatim, like every other mode.

### Done when

- Geometry tests (no checkpoint): row counts, block order incl. audio-before-video inside a video
  reference, clock advancement for each modality (image +1.0; the max() rule), tags, label
  numbering per modality, 2 fps sampling + timestamp strings against hand-computed cases.
- Parity: `audio-vae-encode`, plus `conditioner-ref2va` at full depth (`hidden_states[50]`, like
  fl2va's — same 51-layer truncation trick) on a small synthetic request (1 image + 1 short video
  with sound). Expect rel RMS ≈0.02 / cosine ≥0.999 (fl2va's bar). Vision tower stays bf16.

## Phase C — pipeline, CLI, weights, E2E (issue #10)

- `H3GenerationRequest.references: [H3Reference]`; ref2va branch in `H3Pipeline.generate`
  (same stage order and memory envelope; `transformer_ref` loaded instead of `transformer` when
  references are present — directory name is the only difference).
- `export-quantized` gains `transformer-ref`; prequantized pickup keyed per component name.
- CLI: repeatable ordered `--reference <file>` (modality by content/extension), `enhance` gains
  the six-section Ref2VA variant (`H3EnhanceVariant.ref2va`) — format in the playbook and the
  official skill.
- E2E on the official case (`scripts/readme/reproducible-768p-ref2va-request.sh`, seed 0, 5 s,
  `aspect_ratio: auto`, video ref + voice-timbre audio ref, dialogue in `<d>[English]…</d>`).
  Download the two CDN inputs into `.local-runs/` first — they may not stay up forever.
- Turbo: `minimax_h3_ref2v_turbo_4step_v0.1_bf16` should fold with `merge-lora` unchanged (its
  alpha 8 is handled by metadata-driven scaling). Validate speech intelligibility before
  recommending — the fl2v 4-step failed exactly there; the 8-step fl2v LoRA has no ref2v sibling
  yet.
- Weights: `transformer_ref/*` download (started 2026-08-20 into the models dir). Disk math on the
  SSD before quantized exports; prune a Turbo variant directory if needed.

### Done when

The official case generates end to end, judged on the moving video (identity held from the video
reference, dialogue spoken with the referenced timbre, background music carried over), README
updated (Status + examples), `INSTRUCTIONS.md` line "ref2va out of scope" amended, issues closed
with the evidence, and the knowledge log carries what was learned.

## Traps already paid for (do not rediscover)

- fp32 vision tower LOOKS more accurate and breaks the conditioner at depth (massive activation,
  layer 43): bf16 everywhere in release paths. Isolation probes explicitly fp32.
- The order of random draws is a reproducibility contract (condition noise before video/audio
  noise; one draw per condition, packed order).
- `xcodebuild test` silently skips env-gated smokes; `.local-runs/parity-venv` is the pinned
  parity environment; scratchpads get reaped.
- The weights volume can wedge MLX forever if it unmounts (CPU-delta diagnostic) and can degrade
  to EPERM while mounted (remount fixes it).
- GPU state moves short benchmarks by 10×; anything timed follows
  [gpu-burst-vs-sustained](../pitfalls/gpu-burst-vs-sustained.md) or is not a number.
