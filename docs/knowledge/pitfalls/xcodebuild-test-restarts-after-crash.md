---
okf_version: "0.1"
kind: pitfall
created: 2026-08-20
---

# `xcodebuild test` restarts after a crash and still reports green

## Symptom

A test that crashes the runner deterministically — here a `UInt8` overflow inside a test helper,
`Fatal error: Not enough bits to represent the passed value` — comes back as:

```
✔ Test run with 19 tests in 16 suites passed after 7.887 seconds.
```

with **exit code 0**. Run the same bundle in one process and it dies:

```
$ xcrun xctest .xcodebuild/Build/Products/Debug/MiniMaxH3Tests.xctest
◇ Test decodingBoundsTheSourceAtMaxDuration() started.
Swift/arm64e-apple-macos.swiftinterface:13152: Fatal error: Not enough bits to represent the passed value
$ echo $?
133
```

## Cause

`xcodebuild test` runs the bundle across several worker processes and **relaunches a worker that
dies**, logging one line about it:

```
Restarting after unexpected exit, crash, or test timeout; summary will include totals from previous launches.
```

Each relaunch prints its own `Test run with N tests…` summary, so the run reports several
different totals — 87, then 51, then 19 in this repo's case — and the last one it prints is the
one a `tail`/`grep` picks up. None of them is the whole suite, and a crashing test simply never
appears in any of them.

## The tell

The counts move between runs. If `Test run with N tests` is not the number you expect, or changes
run to run, something died. Two reliable checks:

```bash
# 1. grep the crash markers out of an xcodebuild run
xcodebuild test -scheme minimax-h3 -destination 'platform=macOS' -derivedDataPath .xcodebuild 2>&1 \
  | grep -E "Fatal error|Restarting after unexpected exit|✘"

# 2. or run the bundle in ONE process, where the exit code is trustworthy
xcodebuild build-for-testing -scheme minimax-h3 -destination 'platform=macOS' -derivedDataPath .xcodebuild
xcrun xctest .xcodebuild/Build/Products/Debug/MiniMaxH3Tests.xctest ; echo "exit=$?"
```

The second is the same invocation the checkpoint-backed tier already needs (`xcodebuild test` does
not forward the environment, so `H3_MODELS_DIR` never reaches it) — so the honest way to run the
suite is `xcrun xctest`, with or without the weights, and `xcodebuild test` is the convenience
that can lie.
