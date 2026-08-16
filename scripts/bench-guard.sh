#!/bin/zsh
# bench-guard.sh - Catch performance regressions without a checkpoint.
#
# `minimax-h3 bench` times the denoising step's hot-path primitives on synthetic weights of the
# real shapes, so it runs in seconds and needs none of the 163 GB. This script runs it and compares
# the "primitives total" against the reference stored in docs/knowledge/benchmarks/bench-reference.tsv,
# failing beyond a tolerance. Point it at a dependency bump (mlx-swift is pinned exact for API
# reasons — performance can still move under a patch release) or at your own optimization.
#
# THE HARD PART IS NOT THE MEASUREMENT, IT IS THE MACHINE STATE. The same binary on the same Mac
# varies by up to 10x between a cold GPU and one that has just finished a long job, and a fixed
# five-minute sleep is NOT enough after a heavy run — that is how this script first failed its own
# validation at +20.8 %. See docs/knowledge/pitfalls/gpu-burst-vs-sustained.md.
#
# So the guard calibrates instead of guessing: it repeatedly runs the cheap 3712-token point (a few
# seconds) as a thermometer until that point lands within HALF the tolerance of its own reference,
# and only then — after letting the machine cool from the thermometer itself — measures the
# geometry you asked for. A machine that never settles is reported as such rather than silently
# producing a number.
#
# Usage: scripts/bench-guard.sh [tokens] [tolerance-percent]
#        scripts/bench-guard.sh 8998 20
#        SKIP_THERMOMETER=1 scripts/bench-guard.sh 8998   # only if you know the GPU is cold

set -u
# awk formats "7.9" as "7,9" under a French locale, and a comma makes every later comparison a
# STRING comparison ("7,9" > "20" is true) — that bug made this guard fail its own validation on a
# drift that was inside tolerance. Pin the locale before anything numeric happens.
export LC_ALL=C
cd "$(dirname "$0")/.."

TOKENS=${1:-8998}
TOLERANCE=${2:-20}
THERMOMETER_TOKENS=3712
THERMOMETER_TRIES=${THERMOMETER_TRIES:-8}
THERMOMETER_WAIT=${THERMOMETER_WAIT:-120}
SETTLE_WAIT=${SETTLE_WAIT:-90}
SKIP_THERMOMETER=${SKIP_THERMOMETER:-0}
BIN=.xcodebuild/Build/Products/Release/minimax-h3
REFERENCE=docs/knowledge/benchmarks/bench-reference.tsv

[[ -x $BIN ]] || {
  echo "no Release binary — build first:"
  echo "  xcodebuild -scheme minimax-h3 -configuration Release -destination 'platform=macOS' \\"
  echo "    -derivedDataPath .xcodebuild build"
  exit 2
}

reference_for() {
  awk -v t="$1" '$1 == t && $2 == "qint8" { print $3 }' $REFERENCE
}
measure() {
  $BIN bench --tokens $1 --quant qint8 2>&1 | awk '/primitives total/ { print $3 }'
}
drift_of() {  # measured, expected -> signed percent
  awk -v m="$1" -v e="$2" 'BEGIN { printf "%.1f", 100 * (m - e) / e }'
}
within() {    # drift, tolerance -> exit 0 if inside
  awk -v d="$1" -v t="$2" 'BEGIN { exit (d < -t || d > t) ? 1 : 0 }'
}
not_slower() {  # drift, tolerance -> exit 0 unless the reading is slower than tolerance allows.
  # The thermometer gate is ONE-SIDED on purpose: a reading faster than reference means the
  # reference is stale (an optimization landed), not that the machine is hot. Treating it as hot
  # would loop until timeout and report "the GPU never settled" for a speed-up.
  awk -v d="$1" -v t="$2" 'BEGIN { exit (d > t) ? 1 : 0 }'
}

EXPECTED=$(reference_for $TOKENS)
[[ -n $EXPECTED ]] || {
  echo "no reference for $TOKENS tokens in $REFERENCE — add one, or pick a listed geometry:"
  awk '!/^#/ && NF { print "  " $1 " tokens" }' $REFERENCE
  exit 2
}

BUSY=$(ioreg -r -c IOAccelerator -d 1 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$')
if [[ -n $BUSY ]] && (( BUSY > 25 )); then
  echo "the GPU reads ${BUSY}% busy — something else is using it. Stop it and rerun."
  exit 2
fi

# Thermometer: the cheap point tells us whether the machine is in its reference state.
if (( SKIP_THERMOMETER == 0 )); then
  THERMOMETER_REFERENCE=$(reference_for $THERMOMETER_TOKENS)
  [[ -n $THERMOMETER_REFERENCE ]] || {
    echo "no reference for the thermometer point ($THERMOMETER_TOKENS tokens) in $REFERENCE."
    echo "Add that row, or run with SKIP_THERMOMETER=1 on a GPU you know is cold."
    exit 2
  }
  # Half the tolerance: the thermometer decides whether to TRUST the real measurement, so it has to
  # be stricter than the verdict it gates. Cold readings land within ~1 %, hot ones at +15-20 %.
  THERMOMETER_TOLERANCE=$(awk -v t="$TOLERANCE" 'BEGIN { print (t / 2 < 5) ? 5 : t / 2 }')
  SETTLED=0
  for try in $(seq 1 $THERMOMETER_TRIES); do
    READING=$(measure $THERMOMETER_TOKENS)
    [[ -n $READING ]] || { echo "bench produced no 'primitives total' line"; exit 2; }
    READING_DRIFT=$(drift_of $READING $THERMOMETER_REFERENCE)
    if not_slower $READING_DRIFT $THERMOMETER_TOLERANCE; then
      printf "thermometer: %.1f s vs %.1f s (%+.1f %%) — the machine is in reference state\n" \
        "$READING" "$THERMOMETER_REFERENCE" "$READING_DRIFT"
      SETTLED=1
      break
    fi
    printf "thermometer: %.1f s vs %.1f s (%+.1f %%) — still hot, waiting %ds (try %d/%d)\n" \
      "$READING" "$THERMOMETER_REFERENCE" "$READING_DRIFT" "$THERMOMETER_WAIT" "$try" "$THERMOMETER_TRIES"
    (( try < THERMOMETER_TRIES )) && sleep $THERMOMETER_WAIT
  done
  (( SETTLED == 1 )) || {
    echo "the GPU never settled to its reference state — measuring now would compare thermal"
    echo "state, not code. Leave the machine idle and rerun."
    exit 2
  }
  # The thermometer just ran the GPU at 100 % for several seconds, while every row in the reference
  # table was measured after five idle minutes. Give the machine that state back before measuring,
  # or the guard reports the warming it caused itself as drift.
  echo "settling for ${SETTLE_WAIT}s before the real measurement …"
  sleep $SETTLE_WAIT
fi

MEASURED=$(measure $TOKENS)
[[ -n $MEASURED ]] || { echo "bench produced no 'primitives total' line"; exit 2; }
DRIFT=$(drift_of $MEASURED $EXPECTED)
printf "%s tokens · measured %.1f s · reference %.1f s · drift %+.1f %% (tolerance ±%d %%)\n" \
  "$TOKENS" "$MEASURED" "$EXPECTED" "$DRIFT" "$TOLERANCE"

within $DRIFT $TOLERANCE && { echo "PASS"; exit 0; }
echo "FAIL — the step's primitives moved beyond tolerance while the machine was in its reference"
echo "       state. If the change is intended (an optimization landed), update $REFERENCE with"
echo "       the new cold measurement and note it in docs/knowledge/log.md."
exit 1
