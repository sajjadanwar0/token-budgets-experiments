import csv
import json
import statistics
import sys
from pathlib import Path

def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} not found", file=sys.stderr)
        sys.exit(1)

    summary_path = results_dir / "summary.json"
    csv_path = results_dir / "iterations.csv"

    if not summary_path.exists() or not csv_path.exists():
        print(f"ERROR: results files missing in {results_dir}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    iters = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            iters.append({
                "iter": int(row["iter"]),
                "seed": int(row["seed"]),
                "events": int(row["events"]),
                "violations": int(row["violations"]),
                "max_phi": int(row["max_phi"]),
                "final_phi": int(row["final_phi"]),
                "total_spent": int(row["total_spent"]),
                "total_dropped": int(row["total_dropped"]),
            })

    INITIAL_CAP = 1_000_000

    conservation_errors = []
    for r in iters:
        accounted = r["total_spent"] + r["total_dropped"] + r["final_phi"]
        if accounted != INITIAL_CAP:
            conservation_errors.append({
                "iter": r["iter"],
                "seed": r["seed"],
                "accounted": accounted,
                "expected": INITIAL_CAP,
                "delta": INITIAL_CAP - accounted,
            })

    md = []
    md.append("# Conjecture 1 Stress Test — Empirical Validation of Lemma 2\n")
    md.append("## Headline\n")
    if summary["total_violations"] == 0:
        md.append(f"✅ **Zero Φ-invariant violations** across "
                  f"{summary['iterations']:,} iterations "
                  f"({summary['total_events']:,} events).\n")
        md.append("Strong empirical support for Lemma 2 (safety preservation under "
                  "Tokio scheduling). Conjecture 1 remains formally open; full "
                  "bisimulation refinement requires Iris/RustBelt mechanization.\n")
    else:
        md.append(f"❌ **{summary['total_violations']} violations** across "
                  f"{summary['iterations']:,} iterations. "
                  f"Violating seeds: {summary['violating_seeds']}\n")
        md.append("This falsifies the monotonicity argument as currently stated. "
                  "Required: investigate failing seeds, identify root cause "
                  "(implementation bug vs. proof gap).\n")

    md.append("\n## Test parameters\n")
    md.append(f"- Iterations: {summary['iterations']:,}")
    md.append(f"- Tasks per iteration: {summary['tasks_per_iter']}")
    md.append(f"- Operations per task: {summary['ops_per_task']:,}")
    md.append(f"- Total operations: ~{summary['tasks_per_iter'] * summary['ops_per_task'] * summary['iterations']:,}")
    md.append(f"- Wall-clock: {summary['wall_clock_secs'] / 60:.1f} minutes")

    md.append("\n## Distribution statistics\n")
    if iters:
        events = [r["events"] for r in iters]
        max_phi = [r["max_phi"] for r in iters]
        md.append(f"- Events per iteration:  mean={statistics.mean(events):.0f}, "
                  f"median={statistics.median(events):.0f}, "
                  f"max={max(events):,}, min={min(events)}")
        md.append(f"- max Φ observed (any iter): {max(max_phi):,} "
                  f"(B₀ = {INITIAL_CAP:,}); ratio = {max(max_phi)/INITIAL_CAP:.4f}")
        if max(max_phi) > INITIAL_CAP:
            md.append(f"  ⚠️ max Φ > B₀ — violation present")
        else:
            md.append(f"  ✓ max Φ ≤ B₀ throughout")

    md.append("\n## Conservation check (sanity)\n")
    md.append(f"For each iteration, we verify spent + dropped + final_phi = B₀.")
    if not conservation_errors:
        md.append(f"✓ Conservation holds in all {len(iters):,} iterations.")
    else:
        md.append(f"❌ Conservation violated in {len(conservation_errors)} iterations:")
        for ce in conservation_errors[:10]:
            md.append(f"  - iter={ce['iter']} seed={ce['seed']}: "
                      f"accounted={ce['accounted']}, expected={ce['expected']}, "
                      f"delta={ce['delta']}")

    md.append("\n## Paper update (for §V Empirical stratified soundness)\n")
    if summary["total_violations"] == 0:
        md.append(f"""
> *We additionally validate Lemma~\\ref{{lem:safety-pres}} via an adversarial
> stress test: {summary['iterations']:,} iterations of randomized concurrent
> workloads (each iteration: {summary['tasks_per_iter']} Tokio tasks ×
> {summary['ops_per_task']:,} operations × random
> \\texttt{{spend}}/\\texttt{{split}}/\\texttt{{merge}}/\\texttt{{Drop}}
> interleavings under work-stealing scheduling) record zero violations of
> the $\\Phi(s) \\le B_0$ invariant across approximately
> {summary['tasks_per_iter'] * summary['ops_per_task'] * summary['iterations'] // 1_000_000}M total
> operations. This is empirical falsification testing, not mechanized
> proof: a single violation would invalidate Lemma~\\ref{{lem:safety-pres}};
> the absence of violations across this scale is evidence that the
> monotonicity argument is correct under Tokio's actual scheduling behaviour.
> The corresponding test corpus is reproducible from
> \\texttt{{token-budgets/experiments/conjecture\\_1\\_stress/}} at the
> archived commit.*
""")
    else:
        md.append(f"""
> *Adversarial stress testing of Lemma~\\ref{{lem:safety-pres}}
> ({summary['iterations']:,} iterations) revealed {summary['total_violations']}
> Φ-invariant violations. Failing seeds: {summary['violating_seeds']}. We
> investigate root cause in Appendix~B. This finding has implications for
> the safety-preservation claim and may require revision.*
""")

    output_md = "\n".join(md)
    out_path = results_dir / "paper_update.md"
    with open(out_path, "w") as f:
        f.write(output_md)
    print(f"Wrote {out_path}")
    print()
    print(output_md)


if __name__ == "__main__":
    main()
