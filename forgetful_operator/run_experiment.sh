#!/usr/bin/env bash

set -euo pipefail

N="${N:-30}"
CAP_UC="${CAP_UC:-60}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MAX_OUTPUT="${MAX_OUTPUT:-30}"
MARGIN="${MARGIN:-0.5}"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY env var not set." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "  Forgetful-Operator Experiment"
echo "  N=${N}"
echo "  cap=${CAP_UC} uc"
echo "  max_output_tokens=${MAX_OUTPUT}"
echo "  margin=${MARGIN}"
echo

echo "[1/3] Condition: python_racy (no lock, post-LLM record)"
python3 python_racy.py --n "$N" --cap "$CAP_UC" \
    --max-output-tokens "$MAX_OUTPUT" --margin "$MARGIN" \
    --output "$RESULTS_DIR/python_racy_anthropic.csv"
echo

echo "[2/3] Condition: python_locked (asyncio.Lock + pre-flight reserve)"
python3 python_locked.py --n "$N" --cap "$CAP_UC" \
    --max-output-tokens "$MAX_OUTPUT" --margin "$MARGIN" \
    --output "$RESULTS_DIR/python_locked_anthropic.csv"
echo

echo "[3/3] Condition: rust_affine_split (Budget::split into sub-budgets)"
(
    cd rust_affine
    cargo build --release --quiet
    ./target/release/forgetful_operator_rust_affine \
        --n "$N" --cap "$CAP_UC" \
        --max-output-tokens "$MAX_OUTPUT" --margin "$MARGIN" \
        --output "../$RESULTS_DIR/rust_affine_anthropic.csv"
)
echo

(
    cd rust_compile_fail
    cargo test --quiet 2>&1 | tail -10
)
echo

echo "  Computing statistical summary"
python3 analyze.py --results-dir "$RESULTS_DIR"
echo

echo "Done. Results in $RESULTS_DIR/"
echo "Summary: $RESULTS_DIR/summary.csv"