#!/usr/bin/env bash

set -e

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set" >&2
  exit 1
fi

BUDGET_SPIKE=${BUDGET_SPIKE:-/home/neo/RustroverProjects/budget-spike}
HARNESS=$BUDGET_SPIKE/target/release/tc_live_harness
OUT_DIR=$BUDGET_SPIKE/sweep_results_anthropic_2000

if [ ! -x "$HARNESS" ]; then
  echo "ERROR: harness not found at $HARNESS" >&2
  echo "Build with: cd $BUDGET_SPIKE && cargo build --release --bin tc_live_harness" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

for wl in lang001 clarification arg_hallucination; do
  echo "==== anthropic / $wl / cap=2000 ===="
  "$HARNESS" \
    --provider anthropic --workload "$wl" \
    --runs 10 --cap-uc 2000 \
    --output-csv "$OUT_DIR/tc_rust_anthropic_${wl}_n10.csv"
done

echo
echo "==== aggregate ===="
python3 /home/neo/PycharmProjects/langgraph-harness/effective_utilization.py \
    --csv-glob "$OUT_DIR/tc_rust_anthropic_*_n10.csv" \
    --max-output-tokens 200 \
    --out "$OUT_DIR/utilization_anthropic_2000.csv"

echo
echo "Done. Send back $OUT_DIR/utilization_anthropic_2000.csv"
echo "and the v34 paper will be patched to fill the three Anthropic rows"
echo "in Table XVII."