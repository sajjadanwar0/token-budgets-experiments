from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from dataclasses import dataclass


WILSON_Z = 1.95996398454


@dataclass
class Cell:
    runtime: str
    workload: str
    provider: str
    temperature: float
    n_runs: int = 0
    n_overshoot: int = 0
    cell_has_overshoot: bool = False


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
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
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_overshoot(s: str) -> bool:
    if s is None or s == "":
        return False
    try:
        return float(s) > 0
    except ValueError:
        return False


def aggregate_cells(rows: list[dict], col_map: dict[str, str]) -> list[Cell]:
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
            temperature = 0.0

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
    total_runs = sum(c.n_runs for c in cells)
    total_overshoot = sum(c.n_overshoot for c in cells)
    raw_per_run = wilson_interval(total_overshoot, total_runs)

    n_cells = len(cells)
    n_cells_overshoot = sum(1 for c in cells if c.cell_has_overshoot)
    pooled_per_cell = wilson_interval(n_cells_overshoot, n_cells)

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

    all_files: list[str] = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: glob '{pattern}' matched no files", file=sys.stderr)
        all_files.extend(matches)
    if not all_files:
        print("FATAL: no input files", file=sys.stderr)
        sys.exit(2)

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

        report_blocks.append(
            f"| `{os.path.basename(path)}` "
            f"| {stats['n_runs_total']} "
            f"| {stats['n_cells_total']} "
            f"| [{stats['raw_per_run_wilson_lower']:.3f}, {stats['raw_per_run_wilson_upper']:.3f}] "
            f"| [{stats['pooled_per_cell_wilson_lower']:.3f}, {stats['pooled_per_cell_wilson_upper']:.3f}] "
            f"| [{stats['hybrid_wilson_lower']:.3f}, {stats['hybrid_wilson_upper']:.3f}] "
            f"|\n"
        )

    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in output_rows:
                w.writerow(r)
        print(f"Wrote {args.output}", file=sys.stderr)

    with open(args.report, "w") as f:
        f.writelines(report_blocks)
    print(f"Wrote {args.report}", file=sys.stderr)

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