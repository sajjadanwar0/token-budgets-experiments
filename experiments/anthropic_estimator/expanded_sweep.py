#!/usr/bin/env python3
"""Expanded mid-loop boundary sweep using tc_live_harness from budget-spike.

This wrapper invokes the Rust tc_live_harness binary once per (provider,
workload, cap) cell, asking it to run N runs internally and emit a CSV.
We then aggregate per-cell CSVs into a summary.

The harness accepts these flags (matching budget-spike/bin/tc_live_harness.rs):
    --provider {openai|anthropic|groq|ollama}
    --workload {lang001|clarification|arg_hallucination}
    --runs N
    --cap-uc UC
    --output-csv FILE

Model, pricing, temperature, max_output_tokens are all hardcoded inside
the harness. The Python wrapper does not pass them; it only chooses
provider, workload, cap, and run count per cell.

Provider matrix (4 providers x 3 workloads x 50 runs = 600 mid-loop target):
  1. OpenAI gpt-4o-mini       (paid, ~$0.005 total)
  2. Anthropic claude-haiku-4.5 (paid, ~$0.14 total)
  3. Groq llama-3.3-70b        (paid but very cheap, <$0.02)
  4. Ollama llama3.2:latest    (local, free; 3.2B Q4_K_M variant)

NOTE: delegation_fanout was dropped because the Rust harness only ships
the original three workloads; adding delegation_fanout is a Rust task
beyond the scope of this sweep.

Repository layout assumed:
    ~/RustroverProjects/token-budgets-experiments/   <- this repo
        budget-spike/                                <- harness lives here
            target/release/tc_live_harness
        tools/expanded_sweep.py                      <- this file

Usage:
    # Build the harness:
    cd budget-spike && cargo build --release --bin tc_live_harness
    cd ..

    # Paid APIs:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export GROQ_API_KEY=...

    # Local Ollama: must be running with llama3.2 pulled:
    #   ollama serve   (in another terminal)
    #   ollama pull llama3.2    (creates llama3.2:latest tag)
    #   ollama list             (verify llama3.2:latest is present)

    python tools/expanded_sweep.py --output sweep_results_expanded/

    # Or just re-run Ollama after fixing the model tag:
    python tools/expanded_sweep.py --providers ollama --output sweep_ollama_retry/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ===============================================================
# Provider configurations
# ===============================================================
# Caps are per-workload starting points; tune after a calibration pass.

@dataclass
class ProviderConfig:
    name: str               # must match the harness's --provider values
    is_local: bool = False  # purely for cost reporting
    caps_per_workload: Dict[str, int] = field(default_factory=dict)


PROVIDERS = [
    ProviderConfig(
        name="openai",
        caps_per_workload={
            "lang001":           200,
            "clarification":     200,
            "arg_hallucination": 200,
        },
    ),
    ProviderConfig(
        name="anthropic",
        caps_per_workload={
            "lang001":           2000,
            "clarification":     2000,
            "arg_hallucination": 2000,
        },
    ),
    ProviderConfig(
        name="groq",
        caps_per_workload={
            "lang001":           500,
            "clarification":     500,
            "arg_hallucination": 500,
        },
    ),
    ProviderConfig(
        name="ollama",
        is_local=True,
        caps_per_workload={
            "lang001":           150,
            "clarification":     150,
            "arg_hallucination": 150,
        },
    ),
]

WORKLOADS = ["lang001", "clarification", "arg_hallucination"]
N_RUNS_PER_CELL = 50   # 4 providers x 3 workloads x 50 = 600 runs total


# ===============================================================
# Pre-flight checks
# ===============================================================

def check_ollama_available() -> bool:
    """Verify ollama is running at localhost:11434 and llama3.2 is pulled."""
    try:
        import urllib.request
        with urllib.request.urlopen(
                "http://localhost:11434/api/tags", timeout=3
        ) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            llama_present = any(m.startswith("llama3.2") for m in models)
            if not llama_present:
                print("Ollama is running but llama3.2 not pulled.")
                print("  Run: ollama pull llama3.2")
                return False
            return True
    except Exception as e:
        print(f"Ollama not reachable at localhost:11434 ({e}).")
        print("  Run: ollama serve   (in another terminal)")
        return False


def preflight(providers: List[ProviderConfig]) -> bool:
    """Verify all required prerequisites are present."""
    env_vars = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq":      "GROQ_API_KEY",
    }
    ok = True
    for p in providers:
        if p.name == "ollama":
            if not check_ollama_available():
                ok = False
        else:
            v = env_vars.get(p.name)
            if v and v not in os.environ:
                print(f"Missing env var: {v} (for {p.name})")
                ok = False
    return ok


# ===============================================================
# Cell runner: invoke tc_live_harness once per cell, parse output CSV
# ===============================================================

def categorize_row(row: Dict[str, str]) -> str:
    """Map a harness CSV row to one of:
        pre_flight_refused  - spend failed before any successful API call
        mid_loop_fired      - spend failed AFTER at least one successful call
        completed           - finished without hitting cap
        self_terminated     - max_agent_steps_reached
        api_error           - network or API error

    Distinguishing pre_flight vs mid_loop uses agent_steps:
        compile_time_reservation_refused + agent_steps == 0 -> pre_flight
        compile_time_reservation_refused + agent_steps >= 1 -> mid_loop
    """
    outcome = row["outcome"]
    try:
        steps = int(row["agent_steps"])
    except (KeyError, ValueError):
        steps = 0

    if outcome == "compile_time_reservation_refused":
        return "pre_flight_refused" if steps == 0 else "mid_loop_fired"
    if outcome == "completed_no_cap_hit":
        return "completed"
    if outcome == "max_agent_steps_reached":
        return "self_terminated"
    if outcome.startswith("api_error"):
        return "api_error"
    return "unknown"


def run_cell(
        provider: ProviderConfig,
        workload: str,
        cap_uc: int,
        n_runs: int,
        output_dir: Path,
        harness_path: str,
) -> Dict:
    """Invoke tc_live_harness for one cell (one subprocess call, N internal
    runs), then parse the harness's output CSV into a summary dict."""

    cell_csv = output_dir / f"{provider.name}_{workload}_n{n_runs}.csv"
    summary = {
        "provider": provider.name,
        "workload": workload,
        "cap_uc": cap_uc,
        "n_runs": n_runs,
        "csv_path": str(cell_csv),
        "completed_runs": 0,
        "mid_loop_fired": 0,
        "pre_flight_refused": 0,
        "self_terminated": 0,
        "api_error_runs": 0,
        "overshoot_runs": 0,
        "total_spent_uc": 0,
    }

    print(f"\n[{provider.name} | {workload} | cap={cap_uc} uc | runs={n_runs}]")
    t0 = time.time()

    cmd = [
        harness_path,
        "--provider", provider.name,
        "--workload", workload,
        "--runs",     str(n_runs),
        "--cap-uc",   str(cap_uc),
        "--output-csv", str(cell_csv),
    ]
    # Generous timeout: ollama can be slow, especially on CPU.
    # 50 runs of 30s each = ~25 min worst case; allow 90 min headroom.
    per_cell_timeout = 5400  # 90 min

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=per_cell_timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  CELL TIMEOUT after {per_cell_timeout}s")
        summary["api_error_runs"] = n_runs
        return summary

    elapsed = time.time() - t0
    print(f"  harness finished in {elapsed:.1f}s "
          f"(rc={result.returncode})")

    if result.returncode != 0:
        err = result.stderr[:300] if result.stderr else "(no stderr)"
        print(f"  STDERR: {err}")
        summary["api_error_runs"] = n_runs
        return summary

    # Parse the harness's CSV
    if not cell_csv.exists():
        print(f"  ERROR: harness reported success but CSV missing: {cell_csv}")
        summary["api_error_runs"] = n_runs
        return summary

    with open(cell_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = categorize_row(row)
            if cat == "mid_loop_fired":
                summary["mid_loop_fired"] += 1
            elif cat == "pre_flight_refused":
                summary["pre_flight_refused"] += 1
            elif cat == "self_terminated":
                summary["self_terminated"] += 1
            elif cat == "completed":
                summary["completed_runs"] += 1
            elif cat == "api_error":
                summary["api_error_runs"] += 1

            try:
                overshoot = int(row.get("overshoot_uc", "0"))
                spent     = int(row.get("total_spent_uc", "0"))
            except ValueError:
                overshoot, spent = 0, 0
            if overshoot > 0:
                summary["overshoot_runs"] += 1
            summary["total_spent_uc"] += spent

    # Brief per-cell preview
    print(f"  mid_loop_fired={summary['mid_loop_fired']}, "
          f"pre_flight_refused={summary['pre_flight_refused']}, "
          f"completed={summary['completed_runs']}, "
          f"self_term={summary['self_terminated']}, "
          f"api_err={summary['api_error_runs']}, "
          f"overshoot={summary['overshoot_runs']}")
    return summary


# ===============================================================
# Main driver
# ===============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sweep_results_expanded/",
                        help="Output directory for per-cell CSVs")
    parser.add_argument("--harness",
                        default="../../budget-spike/target/release/tc_live_harness",
                        help="Path to tc_live_harness binary "
                             "(default: ../../budget-spike/target/release/tc_live_harness, "
                             "assuming you run this from the "
                             "token-budgets-experiments root)")
    parser.add_argument("--n-runs", type=int, default=N_RUNS_PER_CELL,
                        help=f"Runs per cell (default {N_RUNS_PER_CELL})")
    parser.add_argument("--providers", nargs="+", default=None,
                        help="Subset of provider names (default: all)")
    parser.add_argument("--workloads", nargs="+", default=None,
                        help="Subset of workloads (default: all)")
    parser.add_argument("--calibration", action="store_true",
                        help="Calibration mode: 5 runs per cell, cheap pass")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness_p = Path(args.harness)
    if not harness_p.exists():
        print(f"Harness binary not found at {args.harness}")
        print("Build from budget-spike:")
        print("  cd budget-spike && "
              "cargo build --release --bin tc_live_harness")
        print(f"Or override the path: --harness /abs/path/to/tc_live_harness")
        sys.exit(1)

    providers = PROVIDERS
    if args.providers:
        providers = [p for p in PROVIDERS if p.name in args.providers]
    workloads = WORKLOADS
    if args.workloads:
        workloads = [w for w in WORKLOADS if w in args.workloads]

    n_runs = 5 if args.calibration else args.n_runs

    if not preflight(providers):
        print("\nPre-flight checks failed; aborting.")
        sys.exit(1)

    total_cells = len(providers) * len(workloads)
    print(f"\n=== Expanded sweep ===")
    print(f"Harness:      {args.harness}")
    print(f"Providers:    {[p.name for p in providers]}")
    print(f"Workloads:    {workloads}")
    print(f"Runs/cell:    {n_runs}")
    print(f"Cells:        {total_cells}")
    print(f"Total runs:   {total_cells * n_runs}")
    print(f"Calibration:  {args.calibration}")

    # Run all cells
    all_summaries: List[Dict] = []
    cell_i = 0
    for provider in providers:
        for workload in workloads:
            cell_i += 1
            cap = provider.caps_per_workload[workload]
            print(f"\n[{cell_i}/{total_cells}]")
            summary = run_cell(
                provider, workload, cap, n_runs, output_dir, args.harness
            )
            all_summaries.append(summary)

    # Aggregate summary CSV
    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    # Aggregate numbers
    total_runs          = sum(s["n_runs"] for s in all_summaries)
    total_mid_loop      = sum(s["mid_loop_fired"] for s in all_summaries)
    total_pre_flight    = sum(s["pre_flight_refused"] for s in all_summaries)
    total_completed     = sum(s["completed_runs"] for s in all_summaries)
    total_self_term     = sum(s["self_terminated"] for s in all_summaries)
    total_overshoot     = sum(s["overshoot_runs"] for s in all_summaries)
    total_api_errors    = sum(s["api_error_runs"] for s in all_summaries)

    print(f"\n=== Sweep summary ===")
    print(f"Total runs:           {total_runs}")
    print(f"Mid-loop fired:       {total_mid_loop} "
          f"({100*total_mid_loop/max(total_runs,1):.1f}%)")
    print(f"Pre-flight refused:   {total_pre_flight}")
    print(f"Completed within cap: {total_completed}")
    print(f"Self-terminated:      {total_self_term}")
    print(f"Overshoot runs:       {total_overshoot}")
    print(f"API error runs:       {total_api_errors}")
    print(f"\nResults in:           {output_dir}/")
    print(f"Per-cell CSVs:        {output_dir}/<provider>_<workload>_n*.csv")
    print(f"Aggregate summary:    {output_dir}/summary.csv")

    # Paper insert text
    paper_blurb = output_dir / "paper_insert.md"
    with open(paper_blurb, "w") as f:
        f.write("## Expanded mid-loop campaign - paper insert\n\n")
        f.write(f"Total runs: {total_runs}\n")
        f.write(f"Mid-loop fired: {total_mid_loop}\n")
        f.write(f"Pre-flight refused: {total_pre_flight}\n")
        f.write(f"Completed: {total_completed}\n")
        f.write(f"Self-terminated: {total_self_term}\n")
        f.write(f"Overshoot: {total_overshoot}\n\n")
        f.write("### Per-cell table\n\n")
        f.write("| Provider | Workload | Cap (uc) | N | "
                "Mid-loop | Pre-flight | Completed | Self-term | "
                "Overshoot | API err |\n")
        f.write("|" + "---|" * 10 + "\n")
        for s in all_summaries:
            f.write(f"| {s['provider']} | {s['workload']} | "
                    f"{s['cap_uc']} | {s['n_runs']} | "
                    f"{s['mid_loop_fired']} | "
                    f"{s['pre_flight_refused']} | "
                    f"{s['completed_runs']} | "
                    f"{s['self_terminated']} | "
                    f"{s['overshoot_runs']} | "
                    f"{s['api_error_runs']} |\n")
        f.write(f"\nReady for paste into the paper's expanded-sweep section.\n")
    print(f"Paper-insert text:    {paper_blurb}")


if __name__ == "__main__":
    main()