---
okf_version: "0.1"
---

# Index

- [quantization-2026-08](benchmarks/quantization-2026-08.md) — fidélité/vitesse/mémoire des 7 modes + validation round-trip préquantisé

- [quantized-linear-weight-dtype](pitfalls/quantized-linear-weight-dtype.md) — jamais asType(weight.dtype) sur un layer quantisable
- [context-ir-prompting](playbooks/context-ir-prompting.md) — format de prompt officiel, le soundscape pilote le niveau audio
- [avassetwriter-track-interleaving](pitfalls/avassetwriter-track-interleaving.md) — feed all AVAssetWriter tracks concurrently
- [negative-axis-swift-array-subscript](pitfalls/negative-axis-swift-array-subscript.md) — normalize MLX axes before Swift array indexing
- [vae-decode-occupancy-2026-08](investigations/vae-decode-occupancy-2026-08.md) — le décode VAE = N passes IDENTIQUES de 1797 tokens, 4,65 TFLOP/s, batchables exactement
