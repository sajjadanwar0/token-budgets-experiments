#!/usr/bin/env bash

set -e

if [ -z "$GROQ_API_KEY" ]; then
  echo "ERROR: GROQ_API_KEY not set" >&2
  exit 1
fi

BUDGET_SPIKE=${BUDGET_SPIKE:-/home/neo/RustroverProjects/budget-spike}
HARNESS=$BUDGET_SPIKE/target/release/tc_live_harness
OUT_DIR=$BUDGET_SPIKE/sweep_results_groq_arg_halluc_retry

if [ ! -x "$HARNESS" ]; then
  echo "ERROR: harness not found at $HARNESS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

for run_id in 1 2 3 4 5 6 7 8 9 10; do
  echo "==== groq / arg_hallucination / cap=540 / run $run_id ===="
  "$HARNESS" \
    --provider groq --workload arg_hallucination \
    --runs 1 --cap-uc 540 \
    --output-csv "$OUT_DIR/tc_rust_groq_arg_hallucination_run${run_id}.csv"
  if [ "$run_id" != "10" ]; then
    sleep 30
  fi
done

HEAD_FILE="$OUT_DIR/tc_rust_groq_arg_hallucination_run1.csv"
OUT_FILE="$OUT_DIR/tc_rust_groq_arg_hallucination_n10.csv"
head -1 "$HEAD_FILE" > "$OUT_FILE"
for run_id in 1 2 3 4 5 6 7 8 9 10; do
    tail -n +2 "$OUT_DIR/tc_rust_groq_arg_hallucination_run${run_id}.csv" \
        | sed "s/^token_capabilities_rust,1,/token_capabilities_rust,${run_id},/" \
        >> "$OUT_FILE"
done

echo
echo "==== aggregate ===="
python3 /home/neo//PycharmProjects/langgraph-harness/effective_utilization.py \
    --csv-glob "$OUT_FILE" \
    --max-output-tokens 200 \
    --out "$OUT_DIR/utilization_groq_arg_halluc_retry.csv"

echo
echo "Done. Send back $OUT_FILE and we patch the §V-M cell."