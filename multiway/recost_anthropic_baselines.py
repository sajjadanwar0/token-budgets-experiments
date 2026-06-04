from __future__ import annotations
import argparse
import csv
import sys
from typing import Optional

HAIKU_TO_SONNET_MULTIPLIER = 3.0

def detect_columns(rows: list[dict]) -> tuple[str, Optional[str], str]:
    if not rows:
        print("FATAL: input CSV has 0 rows", file=sys.stderr)
        sys.exit(2)

    cols = set(rows[0].keys())

    spent_col = None

    for candidate in ("total_spent_uc", "spent_uc", "spend_uc",
                      "cost_uc", "total_cost_uc"):
        if candidate in cols:
            spent_col = candidate
            break

    if spent_col is None:
        print(
            f"FATAL: cannot find spend column in {sorted(cols)}.\n"
            f"Expected one of: total_spent_uc, spent_uc, spend_uc.",
            file=sys.stderr,
        )
        sys.exit(3)

    overshoot_col = None

    for candidate in ("overshoot_uc", "overshoot", "violation_uc"):
        if candidate in cols:
            overshoot_col = candidate
            break

    if overshoot_col is None:
        print(
            f"FATAL: cannot find overshoot column in {sorted(cols)}.",
            file=sys.stderr,
        )
        sys.exit(3)

    runtime_col = None

    for candidate in ("runtime", "framework", "system", "harness"):
        if candidate in cols:
            runtime_col = candidate
            break

    return runtime_col, spent_col, overshoot_col

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Path to the original Anthropic head-to-head CSV")
    parser.add_argument("--output", required=True,
                        help="Path to write the recosted CSV")
    parser.add_argument("--multiplier", type=float,
                        default=HAIKU_TO_SONNET_MULTIPLIER,
                        help=f"haiku-to-sonnet cost ratio "
                             f"(default {HAIKU_TO_SONNET_MULTIPLIER})")
    args = parser.parse_args()

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.input}", file=sys.stderr)

    runtime_col, spent_col, overshoot_col = detect_columns(rows)
    print(f"Schema detected: runtime={runtime_col}, "
          f"spent={spent_col}, overshoot={overshoot_col}", file=sys.stderr)
    print(f"Applying multiplier: {args.multiplier}x", file=sys.stderr)

    cap_col_candidates = ("cap_uc", "cap", "budget_uc")
    cap_col = next((c for c in cap_col_candidates if c in rows[0]), None)
    pct_col_candidates = ("pct_of_cap", "percent_of_cap", "pct_cap")
    pct_col = next((c for c in pct_col_candidates if c in rows[0]), None)
    out_rows = []

    for row in rows:
        try:
            original_spent = int(float(row[spent_col]))
            original_overshoot = int(float(row[overshoot_col]))
        except (KeyError, ValueError) as e:
            print(f"WARNING: skipping malformed row: {e}", file=sys.stderr)
            continue

        recosted_spent = int(round(original_spent * args.multiplier))
        new_row = dict(row)
        new_row["spent_uc_original"] = original_spent
        new_row["spent_uc_recosted"] = recosted_spent
        new_row["recost_multiplier"] = args.multiplier

        if cap_col is not None and row.get(cap_col):
            try:
                cap_uc = int(float(row[cap_col]))
                new_row["overshoot_uc_recosted"] = max(0, recosted_spent - cap_uc)
                if pct_col is not None:
                    new_row["pct_of_cap_recosted"] = (
                        f"{(recosted_spent / cap_uc * 100):.2f}"
                        if cap_uc > 0 else "n/a"
                    )
            except ValueError:
                new_row["overshoot_uc_recosted"] = "?"
        else:
            new_row["overshoot_uc_recosted"] = (
                "?" if original_overshoot == 0 else original_overshoot * args.multiplier
            )

        out_rows.append(new_row)

    fieldnames = list(out_rows[0].keys())

    with open(args.output, "w", newline="") as f:
        f.write(
            "# Recosted Anthropic head-to-head CSV.\n"
            f"# Original input: {args.input}\n"
            f"# Multiplier applied: {args.multiplier} (haiku-to-sonnet, derived\n"
            f"# from identical 1:5 input:output price ratio across both providers).\n"
            f"# Added columns: spent_uc_original, spent_uc_recosted,\n"
            f"# recost_multiplier, overshoot_uc_recosted, pct_of_cap_recosted.\n"
            f"# Original spent_uc and overshoot_uc columns preserved unchanged.\n"
            f"# Algebra: sonnet_cost = (3I + 15O)/Mtok = 3 * (I + 5O)/Mtok = 3 * haiku_cost.\n"
        )
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in out_rows:
            w.writerow(r)

    print(f"\n RECOSTED PER-RUNTIME SUMMARY (multiplier={args.multiplier}x) ",
          file=sys.stderr)

    if runtime_col is None:
        print("(no runtime column; reporting global aggregate only)", file=sys.stderr)
        n = len(out_rows)
        mean_orig = sum(r["spent_uc_original"] for r in out_rows) / n
        mean_new = sum(r["spent_uc_recosted"] for r in out_rows) / n
        os_orig = sum(1 for r in out_rows if int(float(r[overshoot_col])) > 0)
        os_new = sum(1 for r in out_rows
                     if str(r["overshoot_uc_recosted"]).isdigit() and
                     int(r["overshoot_uc_recosted"]) > 0)
        print(f"  N={n}  mean_original={mean_orig:.0f}  mean_recosted={mean_new:.0f}  "
              f"OS_orig={os_orig}/{n}  OS_recosted={os_new}/{n}", file=sys.stderr)
    else:
        by_runtime: dict[str, list[dict]] = {}
        for r in out_rows:
            by_runtime.setdefault(r[runtime_col], []).append(r)
        print(f"{'Runtime':40s} {'N':>4s} {'mean orig (uc)':>16s} "
              f"{'mean recost (uc)':>18s} {'OS orig':>10s} {'OS recost':>12s}",
              file=sys.stderr)
        for runtime, runs in by_runtime.items():
            n = len(runs)
            mean_orig = sum(r["spent_uc_original"] for r in runs) / n
            mean_new = sum(r["spent_uc_recosted"] for r in runs) / n
            os_orig = sum(1 for r in runs
                          if int(float(r[overshoot_col])) > 0)
            os_new = 0
            for r in runs:
                o = r["overshoot_uc_recosted"]
                if isinstance(o, str) and o.isdigit():
                    if int(o) > 0:
                        os_new += 1
                elif isinstance(o, (int, float)) and o > 0:
                    os_new += 1
            print(f"{runtime:40s} {n:>4d} {mean_orig:>16.0f} {mean_new:>18.0f} "
                  f"{os_orig}/{n:>2d}      {os_new}/{n:>2d}", file=sys.stderr)

    print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()