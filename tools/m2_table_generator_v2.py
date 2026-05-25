"""
m2_table_generator_v2.py — corrected for the real CSV schema produced by
your multiway_compare.py's _summarise() function.

CSV columns (from the inspection): runtime, outcome, agent_steps, cap_uc,
total_spent_uc, pct_of_cap, overshoot_uc, structural_undershoot_uc,
wasted_call_cost_uc, per_step.

The per_step column is a stringified list of dicts (LangChain CostTrackingCallback
records). We ignore it for aggregation.

Usage:
    python3 m2_table_generator_v2.py \
      --gpt4o-csv    sweep_results/m2_gpt4o_lang001_cap1500_n30.csv \
      --gpt4o-rust-csv sweep_results_rust/m2_rust_gpt4o_lang001_cap1500_n30.csv \
      --haiku-csv    sweep_results/m2_haiku_lang001_cap2000_n30.csv \
      --haiku-rust-csv sweep_results_rust/m2_rust_haiku_lang001_cap2000_n30.csv \
      --gpt4o-cap    1500 \
      --haiku-cap    2000 \
      --out-latex    ../token-budgets/paper/m2_table.tex \
      --out-summary  ../token-budgets/paper/m2_summary.md

The Rust CSV is OPTIONAL: if you don't pass --gpt4o-rust-csv / --haiku-rust-csv
the script omits the TB-Rust row and you can edit the resulting LaTeX to add
it manually once tc_live_harness has run.
"""

from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(rows):
    """Returns {runtime: {n, overshoot_count, overshoot_rate, wilson_lo,
    wilson_hi, mean_spent_uc, mean_pct_of_cap}}."""
    by_runtime = defaultdict(list)
    for r in rows:
        by_runtime[r["runtime"]].append(r)
    out = {}
    for rt, trials in by_runtime.items():
        n = len(trials)
        if n == 0:
            continue
        overshoot = sum(1 for t in trials if int(t.get("overshoot_uc", 0) or 0) > 0)
        mean_spent = sum(int(t.get("total_spent_uc", 0) or 0) for t in trials) / n
        mean_pct = sum(float(t.get("pct_of_cap", 0) or 0) for t in trials) / n
        ci = wilson_ci(overshoot, n)
        out[rt] = {
            "n": n,
            "overshoot_count": overshoot,
            "overshoot_rate": overshoot / n,
            "wilson_lo": ci[0],
            "wilson_hi": ci[1],
            "mean_spent_uc": mean_spent,
            "mean_pct_of_cap": mean_pct,
        }
    return out


def load_csv(path):
    if path is None:
        return []
    with Path(path).open() as f:
        return list(csv.DictReader(f))


# Ordering matches the new §5 table; cosmetic only.
RUNTIME_LABELS = {
    "tb_rust_impl":               r"Token Budgets (Rust, affine type)",
    "token_capabilities_bytelen": r"TB Python, byte-length+2$\times$ estimator (NEW, M2)",
    "naive_guard":                r"Naive 4-line counter (no library, no refund)",
    "token_capabilities":         r"TB Python, coarse fixed-form estimator (existing)",
    "langgraph_with_guard":       r"LangGraph + AgentGuard callback (control)",
}


def emit_latex(agg_gpt4o, agg_haiku, cap_gpt4o, cap_haiku, out_path):
    rows = []
    for rt, label in RUNTIME_LABELS.items():
        g = agg_gpt4o.get(rt)
        h = agg_haiku.get(rt)
        if g is None and h is None:
            continue
        g_cell = f"{g['overshoot_count']}/{g['n']}" if g else "---"
        h_cell = f"{h['overshoot_count']}/{h['n']}" if h else "---"
        g_ci = f"[{g['wilson_lo']:.3f}, {g['wilson_hi']:.3f}]" if g else "---"
        h_ci = f"[{h['wilson_lo']:.3f}, {h['wilson_hi']:.3f}]" if h else "---"
        rows.append(f"  {label} & {g_cell} & {g_ci} & {h_cell} & {h_ci} \\\\")

    if not rows:
        Path(out_path).write_text("% No data; check CSV paths.\n")
        print(f"Wrote (empty) {out_path}")
        return

    latex = f"""\\begin{{table}}[!t]
\\caption{{\\textbf{{M2 head-to-head: isolating estimator and language from
discipline.}} LANG-001, $N=30$ per cell, $T=0$. Caps put the discipline in
the discriminating regime (admits some calls, refuses the cap-violating
call). Wilson 95\\% intervals on overshoot rate per replica; statistical
conventions of \\S\\ref{{sec:eval}} apply (effective $N$ per cell is
below 30 at $T=0$). The three pre-flight rows differ only in
(language, estimator); the AgentGuard row is the post-call observation
control.}}
\\label{{tab:m2-isolation}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{p{{0.34\\columnwidth}} c c c c}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{\\texttt{{gpt-4o}}, $B_0={cap_gpt4o}$\\,uc}}
 & \\multicolumn{{2}}{{c}}{{\\texttt{{claude-haiku-4-5}}, $B_0={cap_haiku}$\\,uc}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Runtime & Overshoot & 95\\% CI & Overshoot & 95\\% CI \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(latex)
    print(f"Wrote {out_path}")


def emit_summary(agg_gpt4o, agg_haiku, cap_gpt4o, cap_haiku, out_path):
    lines = []
    lines.append("# M2 results summary\n")
    lines.append(f"Caps: gpt-4o @ {cap_gpt4o} uc; haiku @ {cap_haiku} uc. N=30 per cell, T=0.\n")
    lines.append("## Per-runtime aggregates\n")
    lines.append("| Runtime | gpt-4o overshoot | gpt-4o CI | gpt-4o mean spend (uc) | haiku overshoot | haiku CI | haiku mean spend (uc) |")
    lines.append("|---|---|---|---|---|---|---|")
    for rt, label in RUNTIME_LABELS.items():
        g = agg_gpt4o.get(rt)
        h = agg_haiku.get(rt)
        if g is None and h is None:
            continue
        g_cell = f"{g['overshoot_count']}/{g['n']}" if g else "---"
        h_cell = f"{h['overshoot_count']}/{h['n']}" if h else "---"
        g_ci = f"[{g['wilson_lo']:.3f}, {g['wilson_hi']:.3f}]" if g else "---"
        h_ci = f"[{h['wilson_lo']:.3f}, {h['wilson_hi']:.3f}]" if h else "---"
        g_mean = f"{g['mean_spent_uc']:.0f}" if g else "---"
        h_mean = f"{h['mean_spent_uc']:.0f}" if h else "---"
        lines.append(f"| {label} | {g_cell} | {g_ci} | {g_mean} | {h_cell} | {h_ci} | {h_mean} |")

    lines.append("\n## Decision tree\n")
    # Try to detect cases:
    def best(rt):
        return agg_gpt4o.get(rt) or agg_haiku.get(rt)

    rust    = best("tb_rust_impl")
    tb_byte = best("token_capabilities_bytelen")
    tb_coarse = best("token_capabilities")
    ag      = best("langgraph_with_guard")

    rust_zero    = rust    is not None and rust["overshoot_count"] == 0
    byte_zero    = tb_byte is not None and tb_byte["overshoot_count"] == 0
    coarse_pos   = tb_coarse is not None and tb_coarse["overshoot_count"] > 0
    coarse_zero  = tb_coarse is not None and tb_coarse["overshoot_count"] == 0
    ag_pos       = ag      is not None and ag["overshoot_count"] > 0

    # Case A: rust=0, tb_byte=0, tb_coarse>0 (estimator is the variable),
    #         ag>0 (post-call observer overshoots as control).
    # This is the most likely and most paper-strengthening outcome.
    if rust_zero and byte_zero and coarse_pos and ag_pos:
        lines.append("\n**Case A**: Rust and TB-Python-bytelen both 0; TB-Python-coarse overshoots; control overshoots.\n")
        lines.append("=> Use the **Case A wording** from m2-paper-section.tex.\n")
        lines.append("=> ESTIMATOR is the discriminating variable, not the type system. The Rust impl and the\n")
        lines.append("   Python impl with same estimator reach the same outcome. The coarse-estimator\n")
        lines.append("   Python overshoots because A1 fails on its calibration. This isolates two\n")
        lines.append("   contributions of the discipline (estimator soundness; type-system non-bypassability)\n")
        lines.append("   that were tangled in Table 8.\n")
    elif rust_zero and byte_zero and coarse_zero and ag_pos:
        lines.append("\n**Case B**: ALL three pre-flight runtimes hit 0; only the post-call control overshoots.\n")
        lines.append("=> Use the **Case B wording** from m2-paper-section.tex.\n")
        lines.append("=> Pre-flight timing alone is sufficient on this workload. The coarse estimator did NOT\n")
        lines.append("   overshoot in this run despite Table 8's earlier observation; check whether the cap\n")
        lines.append("   was chosen above the coarse estimator's bound. Worth re-running at a tighter cap.\n")
    elif rust_zero and not byte_zero:
        lines.append("\n**Case C**: TB-Python-bytelen overshoots while TB-Rust does not, at otherwise-equal estimator.\n")
        lines.append("=> Use the **Case C wording** from m2-paper-section.tex.\n")
        lines.append("=> Investigate the Python adapter's _byte_length_of_messages serialisation against the\n")
        lines.append("   Rust ByteLengthEstimator. The two must produce identical byte counts on the same\n")
        lines.append("   message stack for this comparison to isolate the type system rather than the\n")
        lines.append("   serialisation logic.\n")
    elif rust is None:
        lines.append("\n**TB-Rust row missing** (no --gpt4o-rust-csv passed).\n")
        lines.append("=> Run `tc_live_harness` separately, point --gpt4o-rust-csv at its output, re-run this script.\n")
        lines.append("=> The Python-side comparison alone (token_capabilities vs token_capabilities_bytelen)\n")
        lines.append("   already isolates ESTIMATOR CHOICE; the Rust comparison adds the language-isolation\n")
        lines.append("   point that R#3/R#4 specifically asked about.\n")
    else:
        lines.append("\n**Result does not match a clean case.** Inspect per-trial CSVs by hand.\n")
        lines.append("If TB-Rust itself overshoots, something is wrong with the harness wiring or the cap is sub-floor;\n")
        lines.append("re-run the smoke test at N=1 first.\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt4o-csv",      required=True,
                    help="Python-harness CSV for gpt-4o (multiway_compare.py output)")
    ap.add_argument("--gpt4o-rust-csv", required=False, default=None,
                    help="Rust-harness CSV for gpt-4o (tc_live_harness output, optional)")
    ap.add_argument("--haiku-csv",      required=True,
                    help="Python-harness CSV for claude-haiku-4-5 (multiway_compare.py output)")
    ap.add_argument("--haiku-rust-csv", required=False, default=None,
                    help="Rust-harness CSV for claude-haiku-4-5 (tc_live_harness output, optional)")
    ap.add_argument("--gpt4o-cap",   type=int, default=1500)
    ap.add_argument("--haiku-cap",   type=int, default=2000)
    ap.add_argument("--out-latex",   required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    g_rows = load_csv(args.gpt4o_csv) + load_csv(args.gpt4o_rust_csv)
    h_rows = load_csv(args.haiku_csv) + load_csv(args.haiku_rust_csv)

    agg_g = aggregate(g_rows)
    agg_h = aggregate(h_rows)

    emit_latex(agg_g, agg_h, args.gpt4o_cap, args.haiku_cap, args.out_latex)
    emit_summary(agg_g, agg_h, args.gpt4o_cap, args.haiku_cap, args.out_summary)


if __name__ == "__main__":
    main()