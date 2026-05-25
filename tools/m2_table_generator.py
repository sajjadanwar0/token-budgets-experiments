"""
m2_table_generator.py

Reads the two M2 CSVs (gpt-4o and Sonnet at the discriminating caps) and
emits:
    1. A LaTeX-ready table (one --out-latex) for paste into §5.
    2. A summary markdown (--out-summary) for human inspection.

Usage:
    python3 m2_table_generator.py \
      --gpt4o-csv  sweep_results/m2_gpt4o_lang001_cap1200_n30.csv \
      --sonnet-csv sweep_results/m2_sonnet_lang001_cap2000_n30.csv \
      --out-latex  token-budgets-paper/m2_table.tex \
      --out-summary token-budgets-paper/m2_summary.md

CSV schema expected (per row, one per trial):
    runtime,provider,model,workload,cap_uc,trial,spent_uc,overshoot_uc,outcome
"""

from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


# Wilson score interval for binomial proportions.
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(rows):
    """Returns {runtime: {'n': int, 'overshoot_count': int, 'mean_spent_uc': float}}."""
    by_runtime = defaultdict(list)
    for r in rows:
        by_runtime[r["runtime"]].append(r)
    out = {}
    for rt, trials in by_runtime.items():
        n = len(trials)
        overshoot = sum(1 for t in trials if int(t["overshoot_uc"]) > 0)
        mean_spent = sum(int(t["spent_uc"]) for t in trials) / n if n else 0
        ci = wilson_ci(overshoot, n)
        out[rt] = {
            "n": n,
            "overshoot_count": overshoot,
            "overshoot_rate": overshoot / n if n else 0.0,
            "wilson_lo": ci[0],
            "wilson_hi": ci[1],
            "mean_spent_uc": mean_spent,
        }
    return out


def load_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


RUNTIME_LABELS = {
    "tb_rust_impl":             r"Token Budgets (Rust, affine type)",
    "tb_python_bytelen":        r"TB Python port, byte-length estimator",
    "naive_guard":              r"Naive 4-line pre-flight guard",
    "langgraph_with_agentguard": r"LangGraph + AgentGuard callback",
}


def emit_latex(agg_gpt4o, agg_sonnet, cap_gpt4o, cap_sonnet, out_path):
    rows = []
    for rt, label in RUNTIME_LABELS.items():
        g = agg_gpt4o.get(rt)
        s = agg_sonnet.get(rt)
        if g is None and s is None:
            continue
        gpt_cell = f"{g['overshoot_count']}/{g['n']}" if g else "---"
        son_cell = f"{s['overshoot_count']}/{s['n']}" if s else "---"
        if g:
            gpt_ci = f"[{g['wilson_lo']:.3f}, {g['wilson_hi']:.3f}]"
        else:
            gpt_ci = "---"
        if s:
            son_ci = f"[{s['wilson_lo']:.3f}, {s['wilson_hi']:.3f}]"
        else:
            son_ci = "---"
        rows.append(f"  {label} & {gpt_cell} & {gpt_ci} & {son_cell} & {son_ci} \\\\")

    latex = f"""\\begin{{table}}[!t]
\\caption{{\\textbf{{M2 head-to-head: pre-flight timing vs.\\ type-system enforcement.}}
LANG-001, $N=30$ per cell, $T=0$. Caps chosen to put the discipline in the
discriminating regime (admits some calls, refuses the cap-violating call).
The three pre-flight rows (TB Rust, TB Python with same estimator, naive
4-line guard) isolate the type-system contribution from pre-flight timing;
the AgentGuard row is the post-call observation control. Wilson 95\\%
intervals on overshoot rate per replica.}}
\\label{{tab:m2-isolation}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{p{{0.36\\columnwidth}} c c c c}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{\\texttt{{gpt-4o}}, $B_0={cap_gpt4o}$\\,uc}}
 & \\multicolumn{{2}}{{c}}{{\\texttt{{claude-sonnet-4-5}}, $B_0={cap_sonnet}$\\,uc}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Runtime & Overshoot & 95\\% CI & Overshoot & 95\\% CI \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    Path(out_path).write_text(latex)
    print(f"Wrote {out_path}")


def emit_summary(agg_gpt4o, agg_sonnet, cap_gpt4o, cap_sonnet, out_path):
    lines = []
    lines.append("# M2 results summary\n")
    lines.append(f"Caps: gpt-4o @ {cap_gpt4o} uc; sonnet @ {cap_sonnet} uc. N=30 per cell, T=0.\n")
    lines.append("## Per-runtime overshoot rates\n")
    lines.append("| Runtime | gpt-4o overshoot | gpt-4o 95% CI | Sonnet overshoot | Sonnet 95% CI |")
    lines.append("|---|---|---|---|---|")
    for rt, label in RUNTIME_LABELS.items():
        g = agg_gpt4o.get(rt)
        s = agg_sonnet.get(rt)
        if g is None and s is None:
            continue
        g_cell = f"{g['overshoot_count']}/{g['n']}" if g else "---"
        s_cell = f"{s['overshoot_count']}/{s['n']}" if s else "---"
        g_ci = f"[{g['wilson_lo']:.3f}, {g['wilson_hi']:.3f}]" if g else "---"
        s_ci = f"[{s['wilson_lo']:.3f}, {s['wilson_hi']:.3f}]" if s else "---"
        lines.append(f"| {label} | {g_cell} | {g_ci} | {s_cell} | {s_ci} |")

    lines.append("\n## Decision tree\n")
    rust = agg_gpt4o.get("tb_rust_impl") or agg_sonnet.get("tb_rust_impl")
    pyth = agg_gpt4o.get("tb_python_bytelen") or agg_sonnet.get("tb_python_bytelen")
    naiv = agg_gpt4o.get("naive_guard") or agg_sonnet.get("naive_guard")
    ag   = agg_gpt4o.get("langgraph_with_agentguard") or agg_sonnet.get("langgraph_with_agentguard")

    rust_zero = rust and rust["overshoot_count"] == 0
    pyth_zero = pyth and pyth["overshoot_count"] == 0
    naiv_zero = naiv and naiv["overshoot_count"] == 0
    ag_pos    = ag and ag["overshoot_count"] > 0

    if rust_zero and pyth_zero and naiv_zero and ag_pos:
        lines.append("\n**Case A**: all three pre-flight runtimes at 0 overshoot; control overshoots as expected.\n")
        lines.append("=> Use the **Case A wording** from m2-paper-section.tex.\n")
        lines.append("=> The cap-respecting outcome does not require compile-time integrity. The type-system\n")
        lines.append("   contribution is non-bypassability under operator error (Forgetful-Operator).\n")
    elif rust_zero and naiv_zero and (pyth is not None) and not pyth_zero:
        lines.append("\n**Case B**: TB Python port overshoots while Rust and naive guard do not.\n")
        lines.append("=> Use the **Case B wording** from m2-paper-section.tex.\n")
        lines.append("=> Investigate the Python port _consumed flag implementation; the Rust affine type\n")
        lines.append("   catches something the Python runtime check does not.\n")
    elif rust_zero and pyth_zero and (naiv is not None) and not naiv_zero:
        lines.append("\n**Case C**: naive guard overshoots while TB-Rust and TB-Python do not.\n")
        lines.append("=> Use the **Case C wording** from m2-paper-section.tex.\n")
        lines.append("=> The TB Python port adds something beyond bare pre-flight checking. Worth\n")
        lines.append("   examining the receipt/refund cycle on cancelled or errored calls.\n")
    else:
        lines.append("\n**Result does not match a clean case.** Inspect per-trial CSVs by hand.\n")
        lines.append("If TB-Rust itself overshoots, something is wrong with the harness wiring;\n")
        lines.append("re-run the smoke test at N=1.\n")

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt4o-csv",   required=True)
    ap.add_argument("--sonnet-csv",  required=True)
    ap.add_argument("--out-latex",   required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--gpt4o-cap",   type=int, default=1200)
    ap.add_argument("--sonnet-cap",  type=int, default=2000)
    args = ap.parse_args()

    g_rows = load_csv(Path(args.gpt4o_csv))
    s_rows = load_csv(Path(args.sonnet_csv))
    agg_g = aggregate(g_rows)
    agg_s = aggregate(s_rows)

    Path(args.out_latex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)

    emit_latex(agg_g, agg_s, args.gpt4o_cap, args.sonnet_cap, args.out_latex)
    emit_summary(agg_g, agg_s, args.gpt4o_cap, args.sonnet_cap, args.out_summary)


if __name__ == "__main__":
    main()