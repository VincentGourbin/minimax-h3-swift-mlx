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

**Also measured** (clean idle benchmarks, 2026-08-05 evening, 3.7k-token steps): qint8
transformer steps are ~1.34x bf16 (41.9s vs 31.2s avg) for a 45% peak-memory cut (34.4 vs
63.2 GB). An earlier "5x slower" reading was interactive-GPU-contention pollution — always
benchmark with NOTHING else running, including the assistant's own parallel work. Cold
prequantized TE load: 30s (the 2.6s figure was file-cache-warm) vs 55.7s bf16.
Recommended 96 GB config: prequantized qint8 text encoder always; transformer bf16 for speed
or qint8 when memory headroom matters.
