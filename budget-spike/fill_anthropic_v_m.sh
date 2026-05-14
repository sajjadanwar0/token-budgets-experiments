#!/usr/bin/env bash
# fill_anthropic_v_m.sh
#
# Runs the patched tc_live_harness against Anthropic at cap=2000 uc to
# populate the three "—" rows in §V-M Table XVII (Effective utilization).
# At cap=540 (the v33 main sweep cap) Anthropic pre-flight-refuses
# every run because the byte-length estimate of the tool-augmented
# payload at $1/Mtok input exceeds the cap. cap=2000 admits at least
# one call per run on lang001 and arg_hallucination (mid-loop refused
# at step 2) and admits the full clarification flow.
#
# Pre-requisites:
#   - patched tc_live_harness binary built from
#     /mnt/user-data/outputs/tc_live_harness.rs (writes per-call detail
#     columns + workload column)
#   - export ANTHROPIC_API_KEY=sk-ant-...
#
# Output: 3 CSVs in sweep_results_anthropic_2000/, each with N=10 runs
# and the per-call detail columns populated. Estimated cost: ~$0.30
# total across 30 runs (most calls cost ~$0.001 each, lang001/arg_halluc
# refuse at step 2 so 1 successful call per run; clarification typically
# completes in 3 calls).

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