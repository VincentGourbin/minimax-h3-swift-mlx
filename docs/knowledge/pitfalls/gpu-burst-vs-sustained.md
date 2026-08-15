---
okf_version: "0.1"
kind: pitfall
created: 2026-08-14
device: Apple M3 Max, 96 GB unified memory
---

# A cold GPU and a hot GPU are two different machines — up to 10×

Measured while investigating the VAE decode (issue #4). The same Release binary, the same shapes,
the same idle-looking machine on AC power, run at four different moments:

| when | s per decoder pass | TFLOP/s |
|---|---|---|
| first run, GPU idle beforehand | 1.14 | 8.43 |
| right after ~9 min of sustained decode | 11.12 | 0.87 |
| a few minutes later | 2.25 | 4.29 |
| again, minutes later | 1.70 | 5.66 |
| **after 5 idle minutes** | **1.139** | **8.46** |

The last row is the point: the machine *recovers* to within three thousandths of the first
measurement, so the swing is a machine state that comes and goes, not code and not noise.

**The cause is NOT established, and a later observation broke the first explanation.** These runs
were originally attributed to the GPU's power/clock state, because spot-checking
`Device Utilization %` between them read 10 % and `pmset -g therm` recorded no thermal warning.
Hours later the same slow readings reappeared — and this time the GPU held **81–97 % for thirty
seconds with nothing of ours running**, driven by `spotlightknowledged` at 97 % CPU: macOS was
indexing the media files that a remounted volume and our own freshly written MP4s had handed it.

So there are at least two mechanisms that produce an identical signature, and three spot samples
are not enough to tell them apart:

- an external GPU consumer, most plausibly Spotlight's media/knowledge indexing — which **our own
  runs summon**, since every generation writes an MP4 and every remount re-triggers indexing;
- the GPU's own clock state under sustained load.

What is certain is the operational consequence, which is the same either way.

## What this invalidates

- **Any bench run back to back with a long GPU job** measures the hot state and is meaningless.
  Both of the sparse-attention A/B runs and the compile A/B in this repo survived only because
  each carried a control sample in the same thermal state.
- **Any short bench used to predict a long phase** overstates the machine. `bench-decode` predicts
  48 s for a decode that measures 87 s; the missing 39 s is entirely clock state. This is the
  opposite direction from the transformer cost map's note that the bench *overestimates* cost by
  21 % — that comparison was made cold-bench against a long sustained run, and got lucky on sign.

## The protocol

0. **Verify the machine is quiet, do not assume it.** Sample `Device Utilization %` for ~30 s with
   nothing of yours running, not three times over ten seconds, and check
   `ps -Aro pid,pcpu,comm | grep -iE "spotlight|mediaanalysis|photoanalysis"`. A single spot check
   at 10 % is exactly what this pitfall showed to be misleading.
1. Benchmark from a **cold GPU**: at least 5 idle minutes before the first point.
2. Cooldown **before every point**, not just the first — the second variant otherwise runs hot
   and loses on thermals rather than on merit.
3. Alternate the order (A/B/B/A) so any residual drift is visible in the spread.
4. Judge a difference only against the intra-variant spread. With the protocol, points land within
   ±2 %; without it, within 10×.
5. To predict a long phase, measure a long phase. Burst-rate primitives extrapolate to a machine
   that does not exist.
