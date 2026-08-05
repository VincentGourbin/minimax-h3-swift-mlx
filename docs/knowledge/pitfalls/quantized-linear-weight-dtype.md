---
okf_version: "0.1"
kind: pitfall
created: 2026-08-05
---

# `.weight.dtype` on a QuantizedLinear is uint32

**Symptom**: quantized transformer produced pure garbage (blocky noise video) while the quantized
text encoder was bit-perfect (RMS identical to 4 decimals) — isolated in minutes with the
`parity dit-block0 --quant` probe.

**Cause**: the checkpoint-faithful pattern `activations.asType(layer.weight.dtype)` (used to
mirror the reference's mixed-precision casts) breaks after `quantize()`: a `QuantizedLinear`'s
`weight` is the PACKED uint32 tensor, so activations were being cast to integers. Every AdaLN
modulation went through that path → all shift/scale/gate garbage.

**Fix**: `computeDType(_ linear:)` helper — `(linear as? QuantizedLinear)?.scales.dtype ??
linear.weight.dtype` — used for every activation cast keyed on a quantizable layer
(`H3Transformer.swift`).

**Verification**: `parity dit-block0 --quant qint8` — max|Δ| 1.41 vs 1.11 unquantized on a 260
scale (qint8 adds almost nothing over bf16 noise).

**Also measured**: MLX quantized matmuls are substantially slower than bf16 GEMMs at DiT
sequence lengths (3.7k rows: ~1m42/step qint8 vs 20.7-35s bf16; exact ratio to re-verify on an
idle machine — interactive GPU load polluted the afternoon numbers) — the kernels target M=1
LLM generation. Quantization
here is a MEMORY/LOAD tool (peak 34 GB vs 63; prequantized files 3.3x smaller), not a speed tool.
Recommended 96 GB config: prequantized qint8 text encoder (one forward, penalty irrelevant) +
full-precision transformer.
