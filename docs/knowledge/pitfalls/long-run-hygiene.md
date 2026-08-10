---
okf_version: "0.1"
kind: pitfall
created: 2026-08-10
---

# Multi-hour runs die of environment, not of code

A 768p MiniMax-H3 clip is a 7-to-8-hour GPU job. Three separate environmental failures cost a
night of compute each before the run itself was ever at fault. All three are silent.

## 1. MLX wedges forever when the model volume disappears

**Symptom**: the process stays alive with its full footprint resident (47 GB) but stops
progressing. `ps` shows it accumulating **~3 minutes of CPU over 28 hours** — a rounding error
away from zero. A `sample` shows the main thread parked in
`mlx::core::eval_impl → Scheduler::wait_for_one → condition_variable::wait`.

**Cause**: the external SSD holding the weights unmounted mid-run. MLX's stream thread never
returns, and `eval()` waits on a condition variable that will never be signalled. There is no
timeout and no error.

**Fix**: none in-process — `kill -9` and relaunch. Prevention is to not touch the volume during
a run.

**Diagnostic that separates wedged from working**: sample the process's CPU time over ~2 minutes.
A healthy GPU-bound H3 step burns ~1-2 % CPU (≈1.4 s per 120 s) while waiting on the GPU; a
wedged one burns ~0.003 %. Both look identical in `top`'s instantaneous %CPU column (0,0) and
both show state `S`, so only the *delta* over time tells them apart.

## 2. The Mac sleeps and the run silently pauses

**Symptom**: a run launched at 00:50 with a measured 16.5 min/step is still at step 4/29 eight
hours later, yet it is demonstrably alive and its CPU time is still ticking.

**Cause**: macOS entered `Maintenance Sleep` repeatedly on battery from ~01:45 and only woke when
the lid was opened at 07:09 (`pmset -g log | grep -E "Entering Sleep|Wake from"`). The run had
accumulated ~1 h 08 of actual compute out of 7 h 35 of wall clock. Note that `ps -o etime`
reports **awake** time, so the process looks young while the clock says otherwise — the
discrepancy between `etime` and wall clock is itself the tell.

**Fix**: tie a power assertion to the run's lifetime — `caffeinate -is -w <pid> &`. It releases
itself when the run exits. A closed lid on battery still forces sleep regardless.

## 3. `/private/tmp` scratchpads get reaped, and venvs fail *upward*

**Symptom**: a parity dump that ran green in the morning fails at import in the afternoon
(`No module named diffusers.modular_pipelines.minimax_h3`), and the same venv that reported
`torch 2.13.0 / transformers 5.14.1` now reports `2.11.0 / 5.10.2`.

**Cause**: the session scratchpad under `/private/tmp` is periodically purged. The diffusers
checkout lost its `.git` and part of its tree; worse, the venv lost its own `site-packages` and —
having been created with system site-packages visible — kept working while **silently resolving
to different library versions**. A parity harness that changes its reference implementation
without saying so invalidates every number it prints.

**Fix**: keep parity tooling durable and pinned outside `/private/tmp` (`.local-runs/parity-venv`,
gitignored, on the internal SSD), and make probes self-contained where the dependency is
incidental — the `conditioner-fl2va` dump now inlines the canvas stretch instead of importing
diffusers, since canvas preparation is already covered bit-exactly by `keyframe-preprocess`.

## Standing recipe for any run over ~1 hour

- Launch detached (`nohup`), never from a tool call that can be killed at a timeout.
- Write the log to the **internal** disk, even when the outputs go to the external one.
- Arm `caffeinate -is -w <pid>`.
- Keep the launch reproducible in a script (`.local-runs/run-*.sh`) so a relaunch is one command.
- Judge health by CPU-time delta, not by instantaneous %CPU.
