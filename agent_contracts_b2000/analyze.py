from __future__ import annotations
import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from math import lgamma, exp



def read_results(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0: return 1.0

    def log_pmf(a, b, c, d):
        return (lgamma(a + b + 1) + lgamma(c + d + 1)
                + lgamma(a + c + 1) + lgamma(b + d + 1)
                - lgamma(a + 1) - lgamma(b + 1)
                - lgamma(c + 1) - lgamma(d + 1) - lgamma(n + 1))

    p_obs = log_pmf(a, b, c, d)
    total = 0.0
    r1, c1 = a + b, a + c
    a_min = max(0, c1 - (n - r1))
    a_max = min(r1, c1)
    for a_alt in range(a_min, a_max + 1):
        b_alt = r1 - a_alt
        c_alt = c1 - a_alt
        d_alt = n - a_alt - b_alt - c_alt
        if d_alt < 0: continue
        p_alt = log_pmf(a_alt, b_alt, c_alt, d_alt)
        if p_alt <= p_obs + 1e-12:
            total += exp(p_alt)
    return min(1.0, total)


def summarize(label: str, rows: list[dict]) -> dict:
    n = len(rows)
    overshoots = sum(1 for r in rows if int(r["cumulative_spend_uc"]) > int(r["cap_uc"]))
    outcomes = Counter(r["outcome"] for r in rows)

    spends = [int(r["cumulative_spend_uc"]) for r in rows]
    steps = [int(r["steps_admitted"]) for r in rows]
    refusals = sum(int(r["pre_flight_refusals"]) for r in rows)

    lo, hi = wilson_ci(overshoots, n)
    return {
        "label": label,
        "n": n,
        "overshoots": overshoots,
        "overshoot_ci_lo": lo,
        "overshoot_ci_hi": hi,
        "outcome_distribution": dict(outcomes),
        "mean_spend_uc": sum(spends) / n if n else 0,
        "max_spend_uc": max(spends) if spends else 0,
        "mean_steps_admitted": sum(steps) / n if n else 0,
        "total_pre_flight_refusals": refusals,
    }


def render(summaries: list[dict]) -> str:
    out = []
    out.append("=" * 75)
    out.append(" Head-to-head comparison: TB-Python vs AC vs TB-Rust at B_0 = 2,000 uc")
    out.append("=" * 75)
    out.append("")

    out.append(f"{'Framework':<22} {'n':>4} {'overshoot':>12} {'95% CI':>22}")
    out.append("-" * 68)
    for s in summaries:
        rate = f"{s['overshoots']}/{s['n']}"
        ci = f"[{s['overshoot_ci_lo']:.3f}, {s['overshoot_ci_hi']:.3f}]"
        out.append(f"{s['label']:<22} {s['n']:>4} {rate:>12} {ci:>22}")
    out.append("")

    out.append(f"{'Framework':<22} {'mean spend (uc)':>16} "
               f"{'max':>8} {'mean steps':>12} {'refusals':>10}")
    out.append("-" * 75)
    for s in summaries:
        out.append(f"{s['label']:<22} {s['mean_spend_uc']:>16.1f} "
                   f"{s['max_spend_uc']:>8} {s['mean_steps_admitted']:>12.2f} "
                   f"{s['total_pre_flight_refusals']:>10}")
    out.append("")

    # Outcome distributions
    out.append("Outcome distributions:")
    for s in summaries:
        out.append(f"  {s['label']}:")
        for outcome, cnt in sorted(s["outcome_distribution"].items(),
                                   key=lambda kv: -kv[1]):
            out.append(f"    {outcome:24s} {cnt:>4d} / {s['n']}")
    out.append("")

    if len(summaries) >= 2:
        out.append("Pairwise Fisher's exact test (on overshoot counts):")
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                a, b = summaries[i], summaries[j]
                p = fisher_exact_2x2(a["overshoots"], a["n"] - a["overshoots"],
                                     b["overshoots"], b["n"] - b["overshoots"])
                out.append(f"  {a['label']:>22}  vs  {b['label']:<22}  p = {p:.4f}")
        out.append("")

    all_zero = all(s["overshoots"] == 0 for s in summaries)
    if all_zero:
        out.append("Interpretation: all frameworks 0/N overshoot at the discriminating cap.")
        out.append("Cap-respecting outcome is achievable by either compile-time integrity")
        out.append("(TB-Rust), runtime monitoring (AC), or a plain counter (TB-Python).")
        out.append("The affine type system's distinguishing contribution is therefore")
        out.append("non-bypassability under operator error (Forgetful-Operator, sec 5.11),")
        out.append("not the cap-respecting outcome itself.")
    else:
        out.append("Interpretation: frameworks DIFFER on overshoot. Investigate the rows")
        out.append("with non-zero overshoot before drawing conclusions about discipline.")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ac", required=True, type=Path)
    p.add_argument("--tb-py", required=True, type=Path)
    p.add_argument("--tb-rs", type=Path, default=None,
                   help="Optional Rust CSV from tc_live_harness")
    p.add_argument("--output", type=Path, default=Path("comparison_summary.txt"))
    args = p.parse_args()

    summaries = []
    summaries.append(summarize("TB-Python (counter)", read_results(args.tb_py)))
    summaries.append(summarize("AC (ResourceMonitor)", read_results(args.ac)))
    if args.tb_rs and args.tb_rs.exists():
        summaries.append(summarize("TB-Rust (affine)", read_results(args.tb_rs)))

    txt = render(summaries)
    print(txt)
    args.output.write_text(txt + "\n")
    print(f"\nWrote summary to {args.output.resolve()}")


if __name__ == "__main__":
    main()