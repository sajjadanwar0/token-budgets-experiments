#!/usr/bin/env python3
"""
verify_agent_contracts_drift.py

Verify that the first 10 outcomes of the N=30 Agent Contracts sweep match
the existing N=10 baseline within tolerance. Run this AFTER the N=30 sweep
completes (it does not invoke the library or the API).

Usage:
    python verify_agent_contracts_drift.py \
        --n10 sweep_results/agent_contracts_lang001_n10.csv \
        --n30 sweep_results/agent_contracts_lang001_n30.csv

Exit codes:
    0 — outcome distributions agree within tolerance; safe to use N=30 in paper
    1 — outcomes diverge by more than tolerance; investigate before publishing

Tolerance: |new_budget_violations - existing_budget_violations| <= 2 out of 10.
"""
from __future__ import annotations

import argparse
import csv
import sys


def load_outcomes(path: str, limit: int | None = None) -> list[str]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    outcomes = [r["outcome"] for r in rows]
    if limit:
        outcomes = outcomes[:limit]
    return outcomes


def count_budget_violations(outcomes: list[str]) -> int:
    return sum(1 for o in outcomes if "budget_violation" in o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n10", required=True,
                    help="Path to original N=10 CSV")
    ap.add_argument("--n30", required=True,
                    help="Path to new N=30 CSV (full 30 rows)")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="Tolerance for first-10 comparison (default 2)")
    args = ap.parse_args()

    n10_outcomes = load_outcomes(args.n10)
    n30_outcomes = load_outcomes(args.n30)

    if len(n10_outcomes) != 10:
        print(f"WARNING: N=10 CSV has {len(n10_outcomes)} rows, expected 10",
              file=sys.stderr)
    if len(n30_outcomes) != 30:
        print(f"WARNING: N=30 CSV has {len(n30_outcomes)} rows, expected 30",
              file=sys.stderr)

    n10_violations = count_budget_violations(n10_outcomes)
    n30_first10_violations = count_budget_violations(n30_outcomes[:10])
    n30_total_violations = count_budget_violations(n30_outcomes)
    n30_overshoot = sum(1 for _ in n30_outcomes if False)  # placeholder
    # Properly count overshoot from the file
    with open(args.n30, newline="") as f:
        n30_rows = list(csv.DictReader(f))
    n30_overshoot = sum(1 for r in n30_rows
                        if int(r.get("overshoot_uc", 0)) > 0)

    print("=== AGENT CONTRACTS DRIFT VERIFICATION ===\n")
    print(f"  N=10 outcomes:               {dict_count(n10_outcomes)}")
    print(f"  N=30 first-10 outcomes:      {dict_count(n30_outcomes[:10])}")
    print(f"  N=30 full outcomes:          {dict_count(n30_outcomes)}")
    print()
    print(f"  budget_violation, N=10:               {n10_violations}/10")
    print(f"  budget_violation, N=30 first-10:      {n30_first10_violations}/10")
    print(f"  budget_violation, N=30 total:         {n30_total_violations}/30")
    print(f"  overshoot, N=30 total:                {n30_overshoot}/30")
    print()

    drift = abs(n10_violations - n30_first10_violations)
    print(f"  Drift (|first-10 - existing|):        {drift}")
    print(f"  Tolerance:                            {args.tolerance}")

    if drift > args.tolerance:
        print(f"\nFAIL: drift exceeds tolerance. The N=30 sweep produced a "
              f"different outcome distribution from the N=10 baseline.\n"
              f"Investigate before publishing the N=30 row in Table 7.\n",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nPASS: outcome distribution agrees within tolerance.")
    print(f"      N=30 is comparable to N=10; safe to publish.")

    # Headline for Table 7
    if n30_overshoot == 0:
        print(f"\n=== TABLE 7 HEADLINE ===")
        print(f"  Agent Contracts row:  0/30 overshoot")
        print(f"  Wilson 95% CI per-run: [0.000, 0.114]")
        print(f"  (matches the TB row's N and CI; remove the 'n=10 v1 supplementary "
              f"harness' footnote)")


def dict_count(outcomes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    return counts


if __name__ == "__main__":
    main()