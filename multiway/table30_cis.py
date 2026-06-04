import csv
import math
import sys
import random
from collections import defaultdict

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # Phi^-1(1 - 0.025)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))

def bootstrap_rate_ci(outcomes, B=10000, alpha=0.05, seed=42):
    if not outcomes:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(outcomes)
    rates = []
    for _ in range(B):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    lo_idx = int(B * alpha / 2)
    hi_idx = int(B * (1 - alpha / 2))

    return (rates[lo_idx], rates[hi_idx])

def bootstrap_diff_ci(outcomes_a, outcomes_b, B=10000, alpha=0.05, seed=42):
    if not outcomes_a or not outcomes_b:
        return (0.0, 0.0)
    rng = random.Random(seed)
    na, nb = len(outcomes_a), len(outcomes_b)
    diffs = []
    for _ in range(B):
        sa = [outcomes_a[rng.randrange(na)] for _ in range(na)]
        sb = [outcomes_b[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(sum(sb) / nb - sum(sa) / na)
    diffs.sort()
    lo_idx = int(B * alpha / 2)
    hi_idx = int(B * (1 - alpha / 2))

    return (diffs[lo_idx], diffs[hi_idx])

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 table30_cis.py path/to/gpt4o_lang001_n10_full.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    by_runtime = defaultdict(list)

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rt = row.get("runtime", "unknown").strip()
            try:
                ov = int(row.get("overshoot_uc", "0") or 0)
            except (ValueError, TypeError):
                ov = 0
            by_runtime[rt].append(1 if ov > 0 else 0)

    print(f"Loaded {sum(len(v) for v in by_runtime.values())} trials across "
          f"{len(by_runtime)} runtimes from {csv_path}")
    print()

    rows = []

    for rt in sorted(by_runtime.keys()):
        outcomes = by_runtime[rt]
        n = len(outcomes)
        k = sum(outcomes)
        rate = k / n if n else 0.0
        wlo, whi = wilson_ci(k, n)
        blo, bhi = bootstrap_rate_ci(outcomes)
        rows.append({
            "runtime": rt, "k": k, "n": n, "rate": rate,
            "wilson_lo": wlo, "wilson_hi": whi,
            "boot_lo": blo, "boot_hi": bhi,
            "outcomes": outcomes,
        })

    print("=" * 88)
    print(f"{'runtime':25s}  {'k/n':>6s}  {'rate':>6s}  {'Wilson 95% CI':>22s}  {'Bootstrap 95% CI':>22s}")
    print("=" * 88)

    for r in rows:
        print(f"{r['runtime']:25s}  {r['k']:3d}/{r['n']:<3d}  "
              f"{r['rate']:6.3f}  [{r['wilson_lo']:5.3f}, {r['wilson_hi']:5.3f}]      "
              f"[{r['boot_lo']:5.3f}, {r['boot_hi']:5.3f}]")
    print()

    tb_row = next((r for r in rows if "token_capabilities" in r["runtime"] or
                                       "token_budgets" in r["runtime"]), None)
    if tb_row is None:
        print("WARNING: no token_capabilities/token_budgets row found; skipping pairwise.")
    else:
        print("=" * 88)
        print(f"Bootstrap 95% CI on (baseline overshoot rate - Token Budgets overshoot rate)")
        print("=" * 88)
        for r in rows:
            if r is tb_row:
                continue
            lo, hi = bootstrap_diff_ci(tb_row["outcomes"], r["outcomes"])
            sig = "**" if lo > 0 or hi < 0 else "  "
            print(f"  {r['runtime']:25s} - token_budgets:  "
                  f"diff = {r['rate'] - tb_row['rate']:+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}] {sig}")
        print()
        print("  ** = 95% CI excludes zero (statistically significant)")
        print()


if __name__ == "__main__":
    main()