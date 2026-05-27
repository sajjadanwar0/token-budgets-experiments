#!/usr/bin/env python3
"""
analyze_condition_e.py — aggregate Condition E results, produce paper-ready Table 14 row.

PRE-ATTACK (BRUTAL REVIEWER VOICE):
> "Your analysis script could produce any summary. Where's the
>  pre-committed comparison against Conditions A-D from the paper?"

DISPOSITION:
This script encodes the Conditions A-D headline numbers from paper
§5.11 Table 14 as constants. The Condition E row is computed from
the harness output CSV. The pairwise comparisons (A-vs-E, B-vs-E,
C-vs-E) are computed with Fisher's exact test, same as the paper's
A-vs-B, A-vs-C, A-vs-D pairs. The output is a paper-ready LaTeX row
plus statistical comparison report.

USAGE:
    python3 analyze_condition_e.py \
        --results condition_e_results.csv \
        --report condition_e_analysis.txt

REQUIREMENTS:
    Python 3.9+ (uses pure stdlib for Fisher's exact via
    scipy.stats.fisher_exact if available, else hand-rolled exact
    binomial test)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ====================================================================
# PRE-COMMITTED paper §5.11 Table 14 baseline numbers
# ====================================================================
# DO NOT MODIFY without explicit paper text alignment.

PAPER_TABLE_14 = {
    "A_python_racy_b0_60":     {"overshoot": 30, "trials": 30, "wilson_lower": 0.886, "wilson_upper": 1.000},
    "B_python_locked_b0_60":   {"overshoot": 0,  "trials": 30, "wilson_lower": 0.000, "wilson_upper": 0.114},
    "C_rust_affine_split_60":  {"overshoot": 0,  "trials": 30, "wilson_lower": 0.000, "wilson_upper": 0.114},
    "D_rust_affine_split_100": {"overshoot": 0,  "trials": 30, "wilson_lower": 0.000, "wilson_upper": 0.114},
}

# ====================================================================
# Wilson 95% CI on a proportion (pure stdlib)
# ====================================================================

def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval, 95% CI by default."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    n = trials
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = (z * ((p * (1-p) / n + z**2 / (4*n**2)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

# ====================================================================
# Fisher's exact test (2x2) — pure stdlib
# ====================================================================

def log_factorial(n: int) -> float:
    """Stirling-aware log-factorial; exact for small n."""
    if n < 0:
        return float("-inf")
    if n < 100:
        # Exact
        result = 0.0
        for i in range(2, n+1):
            import math
            result += math.log(i)
        return result
    # Stirling
    import math
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

def log_binom_coef(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

def fisher_exact_2x2(a: int, b: int, c: int, d: int, alternative: str = "two-sided") -> float:
    """Fisher's exact test on 2x2 contingency table:
                Group 1   Group 2
       Success     a         b
       Failure     c         d
       Returns two-sided p-value.
    """
    import math
    n1 = a + c  # column 1 total (group 1 trials)
    n2 = b + d  # column 2 total
    n_succ = a + b  # row 1 total (total successes)
    n = n1 + n2

    # Probability of observing exactly (a, b, c, d) under null:
    #   P = C(n1, a) * C(n2, b) / C(n, a+b)
    def log_p_table(k: int) -> float:
        # k = successes in group 1; group 2 successes = n_succ - k
        if k < 0 or k > n1 or (n_succ - k) < 0 or (n_succ - k) > n2:
            return float("-inf")
        return (log_binom_coef(n1, k) + log_binom_coef(n2, n_succ - k)
                - log_binom_coef(n, n_succ))

    log_p_obs = log_p_table(a)

    # Sum probabilities of all tables at least as extreme (two-sided)
    p_value = 0.0
    for k in range(0, n1 + 1):
        log_p_k = log_p_table(k)
        if log_p_k == float("-inf"):
            continue
        if log_p_k <= log_p_obs + 1e-12:  # tables as or more extreme
            p_value += math.exp(log_p_k)

    return min(1.0, p_value)

# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path,
                    help="condition_e_results.csv from the Rust harness")
    ap.add_argument("--report", required=True, type=Path,
                    help="Output text report path")
    ap.add_argument("--budget", type=int, default=60,
                    help="B_0 cap in micro-cents (must match harness run)")
    args = ap.parse_args()

    # Load harness results
    with args.results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"ERROR: no rows in {args.results}", file=sys.stderr)
        sys.exit(1)

    # Aggregate by trial
    trial_spend = defaultdict(int)
    trial_outcomes = defaultdict(list)
    for r in rows:
        trial_id = int(r["trial_id"])
        trial_spend[trial_id] += int(r["actual_charge_uc"])
        trial_outcomes[trial_id].append(r["outcome"])

    trials = sorted(trial_spend.keys())
    n_trials = len(trials)
    overshoots = sum(1 for t in trials if trial_spend[t] > args.budget)
    wilson_lo, wilson_hi = wilson_ci(overshoots, n_trials)

    mean_spend = sum(trial_spend.values()) / n_trials
    median_spend = sorted(trial_spend.values())[n_trials // 2]
    max_spend = max(trial_spend.values())

    # Pairwise Fisher exact: E vs A/B/C/D
    pairwise = {}
    for name, baseline in PAPER_TABLE_14.items():
        # 2x2: rows = overshoot/no-overshoot; cols = E / baseline
        a = overshoots
        b = baseline["overshoot"]
        c = n_trials - overshoots
        d = baseline["trials"] - baseline["overshoot"]
        p = fisher_exact_2x2(a, b, c, d, "two-sided")
        pairwise[name] = p

    # Outcome decision (pre-committed in paper §8.3 M7)
    if overshoots == 0:
        outcome = "(i) PARITY CONFIRMED"
        interpretation = (
            "Compile-time integrity (Condition C) and runtime shared-mutex\n"
            "discipline (Condition E) achieve the same 0/30 outcome under\n"
            "matched allocation; the type-system contribution is the\n"
            "non-bypassability the trybuild evidence covers, not the\n"
            "cap-respecting outcome.")
    else:
        outcome = f"(ii) PARITY NOT CONFIRMED ({overshoots}/{n_trials} overshoot)"
        interpretation = (
            "Investigate root cause: (a) operator-discipline error in the\n"
            "Condition E implementation that requires correction and re-run,\n"
            "OR (b) structural finding that Arc<Mutex<Budget>> with tokio\n"
            "scheduling exhibits a failure mode that affine split avoids.\n"
            "Either way, the type-system non-bypassability claim is\n"
            "unaffected (trybuild covers both patterns).")

    # Compose report
    lines = [
        "M7 Condition E — Analysis Report",
        "=" * 60,
        "",
        f"Input:   {args.results}",
        f"Trials:  {n_trials}  (paper-equivalent: 30)",
        f"Budget:  {args.budget} uc  (paper-equivalent: 60 uc for Conditions A-C, 100 for D)",
        "",
        "RESULTS (Condition E: Rust shared Arc<Mutex<Budget>>):",
        f"  Overshoots:      {overshoots}/{n_trials}",
        f"  Wilson 95% CI:   [{wilson_lo:.3f}, {wilson_hi:.3f}]",
        f"  Mean trial spend:   {mean_spend:.1f} uc ({mean_spend/args.budget*100:.1f}% of cap)",
        f"  Median trial spend: {median_spend} uc",
        f"  Max trial spend:    {max_spend} uc",
        "",
        "PAIRWISE FISHER EXACT (two-sided) vs paper §5.11 Table 14:",
    ]
    for name, p in pairwise.items():
        baseline = PAPER_TABLE_14[name]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        lines.append(f"  E ({overshoots}/{n_trials}) vs {name} "
                     f"({baseline['overshoot']}/{baseline['trials']}):  "
                     f"p = {p:.4g}  {sig}")
    lines.append("")
    lines.append(f"OUTCOME: {outcome}")
    lines.append("")
    lines.append("INTERPRETATION (per paper §8.3 M7 pre-registration):")
    for ln in interpretation.split("\n"):
        lines.append(f"  {ln}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("PAPER-READY TABLE 14 ROW (drop into §5.11):")
    lines.append("")
    lines.append(r"% Add this row to Table 14:")
    if overshoots == 0:
        lines.append(
            r"\textbf{E: Rust shared $\mathit{Arc\langle Mutex\langle Budget\rangle\rangle}$ "
            r"(60 uc, pre-flight + lock)} & "
            f"{overshoots}/{n_trials} & "
            f"[{wilson_lo:.3f}, {wilson_hi:.3f}] & "
            f"{mean_spend:.0f} & "
            f"{mean_spend/args.budget*100:.0f}\\% & "
            r"\\")
    else:
        lines.append(
            r"\textbf{E: Rust shared $\mathit{Arc\langle Mutex\langle Budget\rangle\rangle}$ "
            r"(60 uc, pre-flight + lock)} & "
            f"\\textbf{{{overshoots}/{n_trials}}} & "
            f"[{wilson_lo:.3f}, {wilson_hi:.3f}] & "
            f"{mean_spend:.0f} & "
            f"{mean_spend/args.budget*100:.0f}\\% & "
            r"\\")
    lines.append("")
    lines.append("PAPER-READY §8.3 M7 STATUS LINE (prepend to M7 paragraph):")
    lines.append("")
    if overshoots == 0:
        lines.append(
            r"\textbf{EXECUTED [DATE], outcome (i)}: 0/30 overshoot confirms "
            r"parity with Condition B. Allocation-vs-integrity confound closed."
        )
    else:
        lines.append(
            r"\textbf{EXECUTED [DATE], outcome (ii)}: " +
            f"{overshoots}/{n_trials} overshoot. Investigation outcome: "
            r"[fill in: operator-discipline error / structural Arc+tokio finding]. "
            r"Type-system non-bypassability claim unaffected."
        )
    lines.append("")

    report = "\n".join(lines) + "\n"
    args.report.write_text(report)
    print(report)
    print(f"Wrote report to {args.report}")

    # Exit code reflects outcome
    sys.exit(0 if overshoots == 0 else 10)


if __name__ == "__main__":
    main()
