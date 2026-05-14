#!/usr/bin/env bash
# sweep_boundary.sh - Boundary-stress validation driver for §V.F
#
# Runs the Rust tc_live_harness with per-provider caps tuned so the
# discipline must fire MID-LOOP (admitting at least the first call,
# refusing some later one). This addresses the harsh-review concern that
# the original 90-run sweep refused pre-flight on Anthropic, never
# exercising the boundary case the discipline is designed for.
#
# Per-provider caps (chosen to admit step 1 and refuse step 2 on lang001;
# step counts may vary across workloads, but the design intent holds):
#   OpenAI:   80 uc  (typical step-1 spend ~33 uc on lang001)
#   Anthropic: 2000 uc  (typical step-1 spend ~950 uc)
#   Groq:     300 uc  (typical step-1 spend ~181 uc)
#
# Usage:
#   export OPENAI_API_KEY=...
#   export ANTHROPIC_API_KEY=...
#   export GROQ_API_KEY=...
#   ./sweep_boundary.sh
#
# Output: 9 CSVs in sweep_results_boundary/ following the same column
# format as the v29 main sweep, with cap_uc reflecting the per-provider
# boundary cap.

set -euo pipefail

mkdir -p sweep_results_boundary

declare -A CAPS
CAPS[openai]=80
CAPS[anthropic]=2000
CAPS[groq]=300

for prov in openai anthropic groq; do
    cap=${CAPS[$prov]}
    for wl in lang001 clarification arg_hallucination; do
        out="sweep_results_boundary/tc_rust_${prov}_${wl}_n10_cap${cap}.csv"
        echo "=== ${prov} / ${wl} / cap=${cap}uc ==="
        ./target/release/tc_live_harness \
            --provider "$prov" --workload "$wl" \
            --runs 10 --cap-uc "$cap" \
            --output-csv "$out"
    done
done

echo ""
echo "=== Sweep complete. CSVs in sweep_results_boundary/ ==="
ls -la sweep_results_boundary/

echo ""
echo "Quick summary (per cell, mean spend / mean steps / outcome counts):"
python3 - << 'PY'
import csv
import os
from collections import Counter

dir_ = "sweep_results_boundary"
for fname in sorted(os.listdir(dir_)):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(dir_, fname)
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"  {fname:<60} EMPTY")
        continue
    n = len(rows)
    spend_mean = sum(int(r["total_spent_uc"]) for r in rows) / n
    steps_mean = sum(int(r["agent_steps"]) for r in rows) / n
    cap_uc = int(rows[0]["cap_uc"])
    overshoots = [int(r["overshoot_uc"]) for r in rows]
    max_overshoot = max(overshoots)
    outcomes = Counter(r["outcome"] for r in rows)
    pct = (spend_mean / cap_uc) * 100
    print(f"  {fname:<60} cap={cap_uc:5d}uc, spend={spend_mean:5.1f}uc ({pct:5.1f}%), "
          f"steps={steps_mean:.1f}, max_overshoot={max_overshoot}uc, outcomes={dict(outcomes)}")
PY
