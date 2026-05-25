#!/bin/bash

set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITERATIONS="${ITERATIONS:-10000}"
TASKS="${TASKS:-32}"
OPS_PER_TASK="${OPS_PER_TASK:-1000}"
PANIC_PROB="${PANIC_PROB:-0.0}"

echo "================================================================"
echo "Conjecture 1 Stress Test"
echo "================================================================"
echo "Iterations:    $ITERATIONS"
echo "Tasks:         $TASKS"
echo "Ops/task:      $OPS_PER_TASK"
echo "Panic prob:    $PANIC_PROB"
echo ""

cd "$EXP_DIR"

if ! grep -q '^csv =' Cargo.toml; then
    echo "Adding csv dependency to Cargo.toml..."
    sed -i '/^\[dependencies\]/a csv = "1"' Cargo.toml
fi

echo "Building release binary..."
cargo build --release 2>&1 | tail -3
echo ""

echo "Running $ITERATIONS iterations..."
echo "(This may take 30-60 min depending on machine.)"
echo ""

mkdir -p results
./target/release/stress \
    --iterations "$ITERATIONS" \
    --tasks "$TASKS" \
    --ops-per-task "$OPS_PER_TASK" \
    --panic-probability "$PANIC_PROB" \
    --output results \
    --progress-every $(($ITERATIONS / 50))

echo ""
echo "Done."
echo "  - results/iterations.csv: per-iteration data"
echo "  - results/summary.json:   aggregate summary"
echo "  - results/violation_*.json: violation details (if any)"
