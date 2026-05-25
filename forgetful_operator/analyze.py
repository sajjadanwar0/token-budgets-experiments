import argparse
import csv
import math
import os
from typing import List, Tuple


def wilson_95_ci(k: int, n: int) -> Tuple[float, float]:
    """Wilson 95% CI on proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Compute Fisher's exact two-tailed p-value for the 2x2 table:
        [[a, b], [c, d]]
    where the rows are conditions and columns are (overshoot, no-overshoot).

    Returns the two-tailed p-value. For our use case the counts will
    often be at the boundary (e.g. 30/0 vs 0/30), so we use the
    log-factorial direct computation to avoid overflow.
    """
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    def lgfact(x: int) -> float:
        return math.lgamma(x + 1)

    def hyper_logp(x: int) -> float:
        return (
            lgfact(row1) + lgfact(row2) + lgfact(col1) + lgfact(col2)
            - lgfact(n)
            - lgfact(x) - lgfact(row1 - x)
            - lgfact(col1 - x) - lgfact(row2 - (col1 - x))
        )

    obs_logp = hyper_logp(a)
    x_lo = max(0, col1 - row2)
    x_hi = min(col1, row1)
    p_sum = 0.0
    for x in range(x_lo, x_hi + 1):
        lp = hyper_logp(x)
        if lp <= obs_logp + 1e-12:
            p_sum += math.exp(lp)
    return min(1.0, p_sum)


def read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summarise(rows: List[dict], condition: str) -> dict:
    if not rows:
        return {
            "condition": condition,
            "n_trials": 0,
            "overshoots": 0,
            "overshoot_rate": 0.0,
            "wilson_lo": 0.0,
            "wilson_hi": 0.0,
            "mean_spent_uc": 0.0,
            "max_spent_uc": 0,
            "errors": 0,
        }

    n = len(rows)
    overshoots = sum(int(r["overshoot"]) for r in rows)
    rate = overshoots / n
    lo, hi = wilson_95_ci(overshoots, n)
    spends = [int(r["total_spent_uc"]) for r in rows]
    errors = sum(1 for r in rows if r.get("error", "").strip())

    return {
        "condition": condition,
        "n_trials": n,
        "overshoots": overshoots,
        "overshoot_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_spent_uc": sum(spends) / n if n else 0.0,
        "max_spent_uc": max(spends) if spends else 0,
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    conds = [
        ("python_racy",
         os.path.join(args.results_dir, "python_racy_anthropic.csv")),
        ("python_locked",
         os.path.join(args.results_dir, "python_locked_anthropic.csv")),
        ("rust_affine_split",
         os.path.join(args.results_dir, "rust_affine_anthropic.csv")),
    ]

    summaries = []
    for name, path in conds:
        rows = read_csv(path)
        s = summarise(rows, name)
        summaries.append(s)

    # Pairwise Fisher's exact: racy-vs-locked, racy-vs-rust
    p_racy_vs_locked = None
    p_racy_vs_rust = None
    if summaries[0]["n_trials"] > 0 and summaries[1]["n_trials"] > 0:
        a = summaries[0]["overshoots"]
        b = summaries[0]["n_trials"] - a
        c = summaries[1]["overshoots"]
        d = summaries[1]["n_trials"] - c
        p_racy_vs_locked = fisher_exact_2x2(a, b, c, d)
    if summaries[0]["n_trials"] > 0 and summaries[2]["n_trials"] > 0:
        a = summaries[0]["overshoots"]
        b = summaries[0]["n_trials"] - a
        c = summaries[2]["overshoots"]
        d = summaries[2]["n_trials"] - c
        p_racy_vs_rust = fisher_exact_2x2(a, b, c, d)

    # Print table
    print()
    print("Forgetful-Operator Experiment: Summary")
    print("=" * 78)
    fmt = "{:<22} {:>8} {:>11} {:>22} {:>12} {:>10}"
    print(fmt.format("condition", "N", "overshoots",
                     "Wilson 95% CI", "mean spend", "errors"))
    print("-" * 78)
    for s in summaries:
        ci_str = f"[{s['wilson_lo']:.3f}, {s['wilson_hi']:.3f}]"
        print(fmt.format(
            s["condition"],
            s["n_trials"],
            f"{s['overshoots']}/{s['n_trials']}",
            ci_str,
            f"{s['mean_spent_uc']:.1f} uc",
            s["errors"],
        ))
    print("-" * 78)
    if p_racy_vs_locked is not None:
        print(f"  Fisher's exact, racy vs locked: p = {p_racy_vs_locked:.4g}")
    if p_racy_vs_rust is not None:
        print(f"  Fisher's exact, racy vs rust:   p = {p_racy_vs_rust:.4g}")
    print()

    # Write summary CSV
    summary_path = os.path.join(args.results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_trials", "overshoots",
                    "overshoot_rate", "wilson_lo", "wilson_hi",
                    "mean_spent_uc", "max_spent_uc", "errors",
                    "p_vs_racy_fisher"])
        for s in summaries:
            p = None
            if s["condition"] == "python_locked":
                p = p_racy_vs_locked
            elif s["condition"] == "rust_affine_split":
                p = p_racy_vs_rust
            w.writerow([
                s["condition"], s["n_trials"], s["overshoots"],
                f"{s['overshoot_rate']:.4f}",
                f"{s['wilson_lo']:.4f}", f"{s['wilson_hi']:.4f}",
                f"{s['mean_spent_uc']:.2f}", s["max_spent_uc"],
                s["errors"], f"{p:.6g}" if p is not None else "",
            ])
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
