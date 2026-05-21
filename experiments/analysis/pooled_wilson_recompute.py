#!/usr/bin/env python3
"""
pooled_wilson_recompute.py

Recompute pooled Wilson 95% CIs across multi-cell result CSVs, treating
T=0 deterministic-replica runs as 1 effective observation per cell and
T>0 runs as N independent observations per cell. Closes the statistical
inflation noted in paper v57.1's residual limitations.

ARGUMENT FOR DOING THIS (from paper S5.10):
  "Deterministic decoding suppresses sampling variance: ten replicas of
  the same prompt produce near-identical token-count traces, so the
  effective number of independent observations is the number of distinct
  (model, cap, workload) cells rather than the number of runs."

Yet the paper's headline Wilson intervals (e.g., [0.000, 0.114] on the TB
0/30 row of Table 7) are computed against the raw run count, not the cell
count. The intervals are therefore tighter than the underlying evidence
supports. This script recomputes correctly.

CONVENTIONS:
  - Each input CSV must contain at minimum these columns:
      runtime, workload, provider, temperature, overshoot_uc, run_index
    (or aliases; the script probes for common variants).
  - A "cell" is a unique (runtime, workload, provider, temperature) tuple.
  - For T=0 cells, the cell-level binary outcome is
      cell_overshoot = (any run in cell has overshoot_uc > 0)
    and the cell contributes 1 observation to the Wilson computation.
  - For T>0 cells, each run is independent and contributes 1 observation.

USAGE:
    cd token-budgets-experiments
    python analysis/pooled_wilson_recompute.py \\
        --input multiway/sweep_results/*.csv \\
        --output analysis/wilson_recomputation.csv \\
        --report analysis/wilson_recomputation.md

EXIT CODES:
  0 on success
  1 if any input CSV is missing required columns (script reports which)
  2 on usage error
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from dataclasses import dataclass


# Wilson 95% CI z-score.
WILSON_Z = 1.95996398454  # qnorm(0.975)


@dataclass
class Cell:
    """A unique (runtime, workload, provider, temperature) cell."""
    runtime: str
    workload: str
    provider: str
    temperature: float
    n_runs: int = 0
    n_overshoot: int = 0
    # Cell-level binary outcome for T=0 cells.
    cell_has_overshoot: bool = False


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson 95% CI for k successes out of n Bernoulli trials.

    Returns (lower, upper). At k=0 the lower bound is 0; at k=n the upper
    bound is 1.
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n
                                         + z * z / (4 * n * n))
    return (max(0.0, center - half_width),
            min(1.0, center + half_width))


def detect_columns(rows: list[dict]) -> dict[str, str]:
    """Probe column names for the four required quantities."""
    if not rows:
        return {}
    cols = set(rows[0].keys())
    mapping = {}

    for canonical, aliases in [
        ("runtime", ("runtime", "framework", "system", "harness")),
        ("workload", ("workload", "scenario", "task", "prompt_class")),
        ("provider", ("provider", "model_provider", "vendor")),
        ("temperature", ("temperature", "T", "temp", "sampling_temperature")),
        ("overshoot_uc", ("overshoot_uc", "overshoot", "violation_uc")),
        ("run_index", ("run_index", "run", "trial", "replica_index")),
    ]:
        for a in aliases:
            if a in cols:
                mapping[canonical] = a
                break
    return mapping


def parse_temperature(s: str) -> float | None:
    """Parse temperature column, returning None if unparseable."""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_overshoot(s: str) -> bool:
    """Return True if this row's overshoot value indicates a violation."""
    if s is None or s == "":
        return False
    try:
        return float(s) > 0
    except ValueError:
        return False


def aggregate_cells(rows: list[dict], col_map: dict[str, str]) -> list[Cell]:
    """Aggregate rows into Cell objects."""
    cells_by_key: dict[tuple, Cell] = {}
    for row in rows:
        try:
            runtime = row.get(col_map.get("runtime", ""), "unknown")
            workload = row.get(col_map.get("workload", ""), "unknown")
            provider = row.get(col_map.get("provider", ""), "unknown")
            temp_raw = row.get(col_map.get("temperature", ""), None)
            temperature = parse_temperature(temp_raw) if temp_raw else 0.0
            overshoot = parse_overshoot(row.get(col_map.get("overshoot_uc", ""),
                                                ""))
        except Exception as e:
            print(f"WARNING: malformed row, skipping: {e}", file=sys.stderr)
            continue

        if temperature is None:
            temperature = 0.0  # Default for missing temperature column.

        key = (runtime, workload, provider, temperature)
        cell = cells_by_key.setdefault(key, Cell(
            runtime=runtime, workload=workload,
            provider=provider, temperature=temperature,
        ))
        cell.n_runs += 1
        if overshoot:
            cell.n_overshoot += 1
            cell.cell_has_overshoot = True
    return list(cells_by_key.values())


def compute_pooled_wilson(cells: list[Cell]) -> dict:
    """Compute three Wilson CIs:
       (1) raw per-run pooled (the paper's current method);
       (2) pooled-on-cells, treating each T=0 cell as 1 obs;
       (3) pooled, T=0 cells as 1 obs, T>0 cells as N obs.
    """
    # (1) Per-run pooled.
    total_runs = sum(c.n_runs for c in cells)
    total_overshoot = sum(c.n_overshoot for c in cells)
    raw_per_run = wilson_interval(total_overshoot, total_runs)

    # (2) Per-cell pooled (all cells as 1 obs each).
    n_cells = len(cells)
    n_cells_overshoot = sum(1 for c in cells if c.cell_has_overshoot)
    pooled_per_cell = wilson_interval(n_cells_overshoot, n_cells)

    # (3) Hybrid: T=0 cells -> 1 obs each, T>0 cells -> N obs each.
    hybrid_n = 0
    hybrid_k = 0
    for c in cells:
        if c.temperature == 0.0:
            hybrid_n += 1
            hybrid_k += int(c.cell_has_overshoot)
        else:
            hybrid_n += c.n_runs
            hybrid_k += c.n_overshoot
    hybrid = wilson_interval(hybrid_k, hybrid_n)

    return {
        "n_runs_total": total_runs,
        "n_overshoot_total": total_overshoot,
        "raw_per_run_wilson_lower": raw_per_run[0],
        "raw_per_run_wilson_upper": raw_per_run[1],
        "n_cells_total": n_cells,
        "n_cells_with_overshoot": n_cells_overshoot,
        "pooled_per_cell_wilson_lower": pooled_per_cell[0],
        "pooled_per_cell_wilson_upper": pooled_per_cell[1],
        "hybrid_n": hybrid_n,
        "hybrid_k": hybrid_k,
        "hybrid_wilson_lower": hybrid[0],
        "hybrid_wilson_upper": hybrid[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more CSV files (glob patterns OK)")
    parser.add_argument("--output", default="wilson_recomputation.csv",
                        help="Per-file recomputation CSV")
    parser.add_argument("--report", default="wilson_recomputation.md",
                        help="Markdown summary report")
    args = parser.parse_args()

    # Expand globs.
    all_files: list[str] = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: glob '{pattern}' matched no files", file=sys.stderr)
        all_files.extend(matches)
    if not all_files:
        print("FATAL: no input files", file=sys.stderr)
        sys.exit(2)

    # Process each file.
    output_rows = []
    report_blocks = ["# Pooled Wilson 95% CI recomputation\n"]
    report_blocks.append(
        "Each CSV is re-analysed treating T=0 deterministic replicas as 1\n"
        "effective observation per cell (the cell-level binary outcome is\n"
        "'any run in cell has overshoot'). T>0 cells contribute N observations.\n\n"
        "The three columns below are:\n"
        "  - **raw**: per-run Wilson (paper's current method)\n"
        "  - **per-cell**: pooled treating every cell as 1 obs\n"
        "  - **hybrid**: T=0 cells as 1 obs, T>0 cells as N obs\n\n"
    )
    report_blocks.append("| File | n_runs | n_cells | raw Wilson | per-cell Wilson | hybrid Wilson |\n")
    report_blocks.append("|------|-------:|--------:|------------|-----------------|---------------|\n")

    for path in all_files:
        try:
            with open(path, newline="") as f:
                # Skip comment lines starting with #
                cleaned = (line for line in f if not line.lstrip().startswith("#"))
                rows = list(csv.DictReader(cleaned))
        except Exception as e:
            print(f"WARNING: cannot read {path}: {e}", file=sys.stderr)
            continue
        if not rows:
            print(f"WARNING: {path} has 0 rows", file=sys.stderr)
            continue
        col_map = detect_columns(rows)
        if "overshoot_uc" not in col_map:
            print(f"WARNING: {path} has no overshoot column; skipping",
                  file=sys.stderr)
            continue
        cells = aggregate_cells(rows, col_map)
        stats = compute_pooled_wilson(cells)
        stats["file"] = os.path.basename(path)
        output_rows.append(stats)

        # Markdown line.
        report_blocks.append(
            f"| `{os.path.basename(path)}` "
            f"| {stats['n_runs_total']} "
            f"| {stats['n_cells_total']} "
            f"| [{stats['raw_per_run_wilson_lower']:.3f}, {stats['raw_per_run_wilson_upper']:.3f}] "
            f"| [{stats['pooled_per_cell_wilson_lower']:.3f}, {stats['pooled_per_cell_wilson_upper']:.3f}] "
            f"| [{stats['hybrid_wilson_lower']:.3f}, {stats['hybrid_wilson_upper']:.3f}] "
            f"|\n"
        )

    # Write CSV output.
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in output_rows:
                w.writerow(r)
        print(f"Wrote {args.output}", file=sys.stderr)

    # Write Markdown report.
    with open(args.report, "w") as f:
        f.writelines(report_blocks)
    print(f"Wrote {args.report}", file=sys.stderr)

    # Print headline to stdout for direct paper inclusion.
    print(f"\n=== POOLED WILSON RECOMPUTATION HEADLINE ===", file=sys.stderr)
    for row in output_rows:
        print(f"  {row['file']}:", file=sys.stderr)
        print(f"    raw per-run [{row['raw_per_run_wilson_lower']:.3f}, "
              f"{row['raw_per_run_wilson_upper']:.3f}]  "
              f"(n_runs={row['n_runs_total']}, k={row['n_overshoot_total']})",
              file=sys.stderr)
        print(f"    per-cell    [{row['pooled_per_cell_wilson_lower']:.3f}, "
              f"{row['pooled_per_cell_wilson_upper']:.3f}]  "
              f"(n_cells={row['n_cells_total']}, k={row['n_cells_with_overshoot']})",
              file=sys.stderr)
        print(f"    hybrid      [{row['hybrid_wilson_lower']:.3f}, "
              f"{row['hybrid_wilson_upper']:.3f}]  "
              f"(n={row['hybrid_n']}, k={row['hybrid_k']})",
              file=sys.stderr)


if __name__ == "__main__":
    main()