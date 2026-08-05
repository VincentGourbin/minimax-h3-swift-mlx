---
okf_version: "0.1"
---

# Knowledge log

- 2026-08-04: Bundle created; porting spec extracted from diffusers minimax-h3 branch into INSTRUCTIONS.md.
- 2026-08-05: Decode paths of both VAEs validated against real weights (audio: 779 tensors,
  weight-norm merged at load; video: 585 tensors, 7 latents -> 22 frames chunk geometry exact).
  Text encoder (50 layers, 551 tensors) loads in ~56 s from the Lexar SSD and produces
  correctly-shaped unnormalized hidden states. Full t2va pipeline + MP4 mux written.
- 2026-08-05: Pitfall: [negative-axis-swift-array-subscript](pitfalls/negative-axis-swift-array-subscript.md) — MLX negative axes vs Swift Array subscript, crashed tiled VAE decode.
- 2026-08-05: Numeric parity EXACTE vs diffusers: audio VAE max|d| 5e-6, video VAE max|d| 1.3e-5.
- 2026-08-05: E2E t2va WORKS: 512x288/22f/8 steps in 204.7 s on M3 Max 96 GB, valid MP4, frame semantically matches the prompt (fox/snow/forest). Sequential loading held within memory.
- 2026-08-05: Pitfall: [avassetwriter-track-interleaving](pitfalls/avassetwriter-track-interleaving.md) — sequential track feeding cross-blocks the MP4 muxer past ~2-3 s; overnight 960x544 run lost to it. Export now streams both tracks concurrently; mux command + raw safetensors dump added as insurance.
- 2026-08-05: Quality run 960x544/124f/29 steps: 2h10 denoise + ~25 min loads/decode. Photorealistic result, fixed exporter+raw-dump insurance validated in production. Audio quiet (max -36 dB) — to assess by ear.
- 2026-08-05: Profiler branché (rapport + Perfetto). Ventilation run léger 960x544/22f/10s : loads 31% (SSD-bound), denoise 45% (GPU 81%), VAE decode 23% avec GPU 17% -> décode passé en fp16 (recette autocast officielle), parité 0.011 max. Fit lin+quad sur les 2 runs : attention ~75% du step à 17k tokens.
- 2026-08-05: fp16 VAE decode validé en prod : 104s -> 26.7s (3.9x), GPU 17%->49%, total run léger 7m41 -> 5m55. Frame OK.
- 2026-08-05: Playbook: [context-ir-prompting](playbooks/context-ir-prompting.md) — audio quasi muet expliqué : H3 attend le format Context-IR (sections integrated_multimodal_description / overall_soundscape / non_diegetic_music). A/B en cours (fox-soundscape).
- 2026-08-05: Enquete audio CLOSE : le niveau suit fidèlement la scène (batterie 0 dBFS vs renard-neige -30 dB), pipeline innocenté par parité, canvas <768 coûte ~6 dB. --normalize-audio ajouté (generate/mux, pic -3 dBFS).
- 2026-08-05: Quantisation livrée : H3Quantization qint8/qint6/int4 + exclusions MiniMax, export préquantisé (export-quantized) avec pickup auto. TE qint8 préquantisé : chargement 55.7s -> 2.6s (x21), fichier 52 -> 27.5 Go, fidélité parfaite. Bug uint32 (asType(weight.dtype) sur QuantizedLinear) attrapé par le harnais de parité, corrigé (computeDType). Steps quantisés nettement plus lents (kernels MLX M=1) : quantisation = outil mémoire/chargement ; config 96 Go recommandée = TE qint8 préquantisé + transformer bf16. Benchmarks step à refaire machine idle.
