import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import math

PAPER_TABLE_14 = {
    "A_python_racy_b0_60":     {"overshoot": 30, "trials": 30, "wilson_lower": 0.886, "wilson_upper": 1.000},
    "B_python_locked_b0_60":   {"overshoot": 0,  "trials": 30, "wilson_lower": 0.000, "wilson_upper": 0.114},
    "C_rust_affine_split_60":  {"overshoot": 0,  "trials": 30, "wilson_lower": 0.000, "wilson_upper": 0.114},
    "D_rust_affine_split_100": {"overshoot": 0,  "trials": 30, "wilson_lower": 0.000, "wilson_upper": 0.114},
}

def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    n = trials
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = (z * ((p * (1-p) / n + z**2 / (4*n**2)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def log_factorial(n: int) -> float:
    if n < 0:
        return float("-inf")
    if n < 100:
        # Exact
        result = 0.0
        for i in range(2, n+1):
            import math
            result += math.log(i)
        return result

    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

def log_binom_coef(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

def fisher_exact_2x2(a: int, b: int, c: int, d: int, alternative: str = "two-sided") -> float:
    n1 = a + c
    n2 = b + d
    n_succ = a + b
    n = n1 + n2

    def log_p_table(k: int) -> float:
        if k < 0 or k > n1 or (n_succ - k) < 0 or (n_succ - k) > n2:
            return float("-inf")
        return (log_binom_coef(n1, k) + log_binom_coef(n2, n_succ - k)
                - log_binom_coef(n, n_succ))

    log_p_obs = log_p_table(a)
    p_value = 0.0

    for k in range(0, n1 + 1):
        log_p_k = log_p_table(k)
        if log_p_k == float("-inf"):
            continue
        if log_p_k <= log_p_obs + 1e-12:
            p_value += math.exp(log_p_k)

    return min(1.0, p_value)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path,
                    help="condition_e_results.csv from the Rust harness")
    ap.add_argument("--report", required=True, type=Path,
                    help="Output text report path")
    ap.add_argument("--budget", type=int, default=60,
                    help="B_0 cap in micro-cents (must match harness run)")
    args = ap.parse_args()

    with args.results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"ERROR: no rows in {args.results}", file=sys.stderr)
        sys.exit(1)

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

    pairwise = {}

    for name, baseline in PAPER_TABLE_14.items():
        a = overshoots
        b = baseline["overshoot"]
        c = n_trials - overshoots
        d = baseline["trials"] - baseline["overshoot"]
        p = fisher_exact_2x2(a, b, c, d, "two-sided")
        pairwise[name] = p

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

    sys.exit(0 if overshoots == 0 else 10)

if __name__ == "__main__":
    main()
