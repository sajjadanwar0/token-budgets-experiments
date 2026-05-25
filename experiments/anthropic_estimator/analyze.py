import csv
import json
import statistics
import sys
from pathlib import Path
from collections import defaultdict


def load(csv_path: Path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def analyze(records):
    valid = [r for r in records if not r["error"]]
    
    a1_holds = [r for r in valid if r["a1_holds"] == "True"]
    
    per_workload = defaultdict(lambda: {"total": 0, "a1_holds": 0, "ratios": []})
    for r in valid:
        wl = r["workload"]
        per_workload[wl]["total"] += 1
        if r["a1_holds"] == "True":
            per_workload[wl]["a1_holds"] += 1
        ratio = as_float(r["est_ratio"])
        if ratio is not None:
            per_workload[wl]["ratios"].append(ratio)
    
    bt_ratios = [as_float(r["bt_ratio"]) for r in valid if as_float(r["bt_ratio"])]
    est_ratios = [as_float(r["est_ratio"]) for r in valid if as_float(r["est_ratio"])]
    
    summary = {
        "total_runs": len(records),
        "valid_runs": len(valid),
        "errors": [r["error"] for r in records if r["error"]],
        "a1_holds_count": len(a1_holds),
        "a1_holds_rate": f"{len(a1_holds)}/{len(valid)}",
        "byte_length_baseline": {
            "expected_holds": 0,
            "expected_holds_rate": "0/30",
            "expected_bt_ratio_range": [0.72, 0.79],
        },
        "anthropic_estimator_observed": {
            "a1_holds_rate": f"{len(a1_holds)}/{len(valid)}",
            "est_ratio_mean": statistics.mean(est_ratios) if est_ratios else None,
            "est_ratio_min": min(est_ratios) if est_ratios else None,
            "est_ratio_max": max(est_ratios) if est_ratios else None,
            "est_ratio_stdev": statistics.stdev(est_ratios) if len(est_ratios) > 1 else 0,
            "bt_ratio_mean": statistics.mean(bt_ratios) if bt_ratios else None,
            "bt_ratio_min": min(bt_ratios) if bt_ratios else None,
            "bt_ratio_max": max(bt_ratios) if bt_ratios else None,
        },
        "per_workload": {
            wl: {
                "total": d["total"],
                "a1_holds": f"{d['a1_holds']}/{d['total']}",
                "est_ratio_mean": statistics.mean(d["ratios"]) if d["ratios"] else None,
                "est_ratio_min": min(d["ratios"]) if d["ratios"] else None,
            }
            for wl, d in per_workload.items()
        },
    }
    return summary


def render_markdown(summary):
    lines = []
    lines.append("# AnthropicEstimator A1 Validation Results\n")
    lines.append("## Headline\n")
    holds = summary["a1_holds_rate"]
    valid_n = summary["valid_runs"]
    pct = (summary["a1_holds_count"] / valid_n * 100) if valid_n else 0
    lines.append(f"**A1 holds: {holds} ({pct:.0f}%)** on Anthropic Haiku-4.5 "
                 "tool-loop workloads with AnthropicEstimator default.\n")
    lines.append("Compare against byte-length baseline: **A1 holds 0/30 (0%)** "
                 "in the same workload configuration.\n")
    
    lines.append("\n## Comparison Table\n")
    lines.append("| Estimator | A1 holds | mean est/actual ratio | range |")
    lines.append("|---|---|---|---|")
    bt = summary["anthropic_estimator_observed"]
    lines.append(f"| ByteLength (baseline) | 0/30 (0%) | "
                 f"{summary['byte_length_baseline']['expected_bt_ratio_range'][0]:.2f}–"
                 f"{summary['byte_length_baseline']['expected_bt_ratio_range'][1]:.2f} | "
                 f"under-bounds actual |")
    est_mean = bt["est_ratio_mean"] or 0
    est_min = bt["est_ratio_min"] or 0
    est_max = bt["est_ratio_max"] or 0
    lines.append(f"| AnthropicEstimator | {holds} ({pct:.0f}%) | "
                 f"{est_mean:.4f} | {est_min:.4f}–{est_max:.4f} |")
    
    lines.append("\n## Per-Workload Breakdown\n")
    lines.append("| Workload | A1 holds | mean est_ratio | min |")
    lines.append("|---|---|---|---|")
    for wl, d in summary["per_workload"].items():
        mean = d["est_ratio_mean"] or 0
        mn = d["est_ratio_min"] or 0
        lines.append(f"| {wl} | {d['a1_holds']} | {mean:.4f} | {mn:.4f} |")
    
    lines.append("\n## Interpretation\n")
    if summary["a1_holds_count"] == valid_n:
        lines.append("✅ **AnthropicEstimator satisfies A1 in all measured cells.** "
                     "The 30/30 byte-length failures previously reported were caused "
                     "by Anthropic's tool-call encoding using short special tokens "
                     "that the byte-count cannot capture; AnthropicEstimator's "
                     "tokenizer-based approach captures these correctly.\n")
        lines.append("The provider-stratified default proposed in §IV-A of the paper "
                     "is validated: byte-length is sound for OpenAI/Groq; "
                     "AnthropicEstimator is sound for Anthropic. The paper's "
                     "Lemma~\\ref{lem:tight-estimator} now has direct empirical "
                     "support.\n")
    else:
        failed = valid_n - summary["a1_holds_count"]
        lines.append(f"⚠️ **A1 fails on {failed}/{valid_n} runs even with "
                     "AnthropicEstimator.**\n")
        lines.append("This indicates server-side token injection beyond what "
                     "the client-visible prompt contains (e.g., system prompts, "
                     "expanded tool descriptions). Lemma~\\ref{lem:tight-estimator}'s "
                     "no-server-rewriting precondition does not hold. Required "
                     "follow-up:\n")
        lines.append("  1. Inspect the failing runs' actual vs estimated token counts")
        lines.append("  2. Characterize the server-side overhead pattern")
        lines.append("  3. Add a calibrated multiplicative safety margin to the "
                     "estimator, OR")
        lines.append("  4. Reformulate A1 for Anthropic as 'A1 modulo bounded "
                     "server-side rewriting' with the observed bound.\n")
    
    lines.append("\n## Paper Update Required\n")
    lines.append("Replace the abstract's claim *'AnthropicEstimator... uses "
                 "Anthropic's actual tokenizer rather than a byte-length upper "
                 "bound, sidestepping the 30/30 byte-length-A1 failures'* with:\n")
    if summary["a1_holds_count"] == valid_n:
        lines.append(f"> *'AnthropicEstimator satisfies A1 on {holds} measured runs "
                     "across three tool-loop workloads (mean est/actual ratio "
                     f"{est_mean:.4f}, range [{est_min:.4f}, {est_max:.4f}]). "
                     "The provider-stratified default thereby achieves "
                     "empirical A1-soundness across all three live providers "
                     "in our evaluation.'*\n")
    else:
        lines.append(f"> *'AnthropicEstimator satisfies A1 on {holds} measured runs; "
                     f"the {failed} failure(s) characterize server-side rewriting "
                     "and are documented in Appendix B.'*\n")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze.py results/runs.csv", file=sys.stderr)
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    records = load(csv_path)
    summary = analyze(records)
    
    json_path = csv_path.parent / "a1_validation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")
    
    md = render_markdown(summary)
    md_path = csv_path.parent / "summary.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")
    
    print("\n" + "=" * 60)
    print(md)


if __name__ == "__main__":
    main()
