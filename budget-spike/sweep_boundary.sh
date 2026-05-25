#!/usr/bin/env bash

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
