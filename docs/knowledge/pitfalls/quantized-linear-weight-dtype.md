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

**Also measured** (clean idle benchmarks, 2026-08-06, 3.7k-token steps): quantized steps are
bf16-speed or better (qint8 28.0s / mxfp8 24.5s / int4 23.6s vs bf16 31.2s) — TWO earlier
readings (5x, then 1.34x) were pollution: benchmark with NOTHING else running, and expect a
first-step kernel-compile spike. A second sibling trap: mxfp/nvfp `scales` are uint8 exponents,
so computeDType must only trust floating scales dtypes. Affine beats microscaling on fidelity
here (qint8 1.41 / mxfp8 5.16 vs bf16 noise 1.11). Recommended default: prequantized qint8 for
both components (fidelity ~transparent, -45% peak memory, 39.7s transformer load).
