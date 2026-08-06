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
- 2026-08-05: Parité text-layer0 OK (max 0.23% rel = bruit bf16) — les 5 sondes vertes. Gotcha harnais : une couche décodeur HF standalone ne s'applique PAS de masque causal (attention_mask=None → bidirectionnel) ; construire le masque 4D additif explicitement dans les dumps.
- 2026-08-05 (soir): Benchs IDLE propres: step qint8 x1.34 vs bf16 (41.9s vs 31.2s, 3.7k tokens), pic 34.4 vs 63.2 Go. Le x5 d'hier = pollution par charge GPU interactive. TE préquantisé à froid: 30s (2.6s = cache disque chaud). Méthodo: benchmark = rien d'autre en parallèle, y compris moi.
- 2026-08-06: Tous les modes MLX câblés (affine + mxfp8/mxfp4/nvfp4 ; piège #2 : scales uint8 des formats mx dans computeDType). Benchs propres : steps quantisés >= vitesse bf16 (qint8 28.0s, int4 23.6s vs bf16 31.2s), fidélité affine > microscaling. Nouveau défaut recommandé : qint8 préquantisé partout. PSNR same-seed = divergence de trajectoire, pas qualité (frames int4 impeccables).
- 2026-08-06: Round-trip préquantisé validé bit-exact pour int4/mxfp8/nvfp4 (layouts avec/sans biases gérés). Benchmark complet publié: docs/knowledge/benchmarks/quantization-2026-08.md. RMS TE mxfp8 dérive (81.4 vs 106.7) — confirmation indépendante affine > microscaling ici.
- 2026-08-06: Caveat mémoire ajouté au bench (relevé par Vincent): les pics profiler sont échantillonnés aux frontières de phase — le transitoire bf16 62 Go du chargement à la volée est invisible ; seul le préquantisé tient sur 32-64 Go. Piste: quantisation par shard au chargement. Le comparatif visuel loggue le wall-time par mode (script sans --profile, assumé — les steps sont couverts par les benchs dédiés).
- 2026-08-06: Comparatif visuel 7 modes livré (5.2s/124f/20σ, même seed): identité préservée en affine 8/6-bit, échantillons divergents (propres) en 4-bit/mx, mxfp4 le plus mou. Wall-times equivalents (90-104 min ±bruit diurne). Mosaïque + planche-contact publiées dans docs/examples.
