from __future__ import annotations
import argparse, csv, math, random, statistics
from pathlib import Path

def load_pair_model(csv_path: Path, cost_col: str, reservation_col: str):

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = []
    a1_viol = 0
    for r in rows:
        rv = r.get(reservation_col, "").strip()
        av = r.get(cost_col, "").strip()
        if rv and av:
            try:
                res, act = float(rv), float(av)
            except ValueError:
                continue
            pairs.append((res, act))
            if res < act:
                a1_viol += 1
    if not pairs:
        raise SystemExit(f"No numeric (reservation, actual) pairs from "
                         f"{reservation_col!r}/{cost_col!r} in {csv_path}")
    import statistics as _s
    print(f"  PAIRED model: {len(pairs)} real (reservation, actual) pairs "
          f"(mean actual {_s.mean(a for _, a in pairs):.1f} uc, "
          f"mean reservation {_s.mean(r for r, _ in pairs):.1f} uc)")
    if a1_viol:
        print(f"  NOTE: {a1_viol}/{len(pairs)} real rows have reservation<actual "
              f"(genuine A1 violations in the data; these are an A1 finding, "
              f"not an A7 artifact)")
    else:
        print(f"  A1 holds in data: reservation >= actual on all "
              f"{len(pairs)} rows")
    return lambda rng: rng.choice(pairs)


def load_cost_model(csv_path: Path | None, cost_col: str,
                    default_mean: float, default_sd: float):
    if csv_path is not None:
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        costs = []
        for r in rows:
            v = r.get(cost_col, "").strip()
            if v:
                try:
                    costs.append(float(v))
                except ValueError:
                    pass
        if not costs:
            raise SystemExit(f"No numeric values in column {cost_col!r} of {csv_path}")
        sd = statistics.pstdev(costs)
        print(f"  cost model: independent-draw bootstrap from {len(costs)} "
              f"real costs (mean {statistics.mean(costs):.1f} uc, sd {sd:.1f} uc)")
        if sd > 1.0:
            print(f"  WARNING: cost has nonzero variance (sd={sd:.1f}); the "
                  f"independent-draw model will produce spurious k=1 overshoot. "
                  f"Use --reservation-col for the correct paired model.")
        return lambda rng: rng.choice(costs)
    else:
        print(f"  cost model: synthetic truncated-normal "
              f"(mean {default_mean} uc, sd {default_sd} uc)")
        def sample(rng):
            c = rng.gauss(default_mean, default_sd)
            return max(1.0, c)
        return sample


def run_trial(cap: float, k: float, margin: float, sampler, rng,
              reconcile_every: int | None, paired: bool) -> dict:
    remaining_ledger = cap
    true_spend = 0.0
    reported_spend = 0.0
    steps = 0
    max_steps = 10_000

    while steps < max_steps:
        if paired:
            reservation, c_true = sampler(rng)
        else:
            est = sampler(rng)
            reservation = est * margin
            c_true = sampler(rng)
        if reservation > remaining_ledger:
            break
        true_spend += c_true
        c_reported = c_true / k
        reported_spend += c_reported
        remaining_ledger -= c_reported
        steps += 1

        if reconcile_every and steps % reconcile_every == 0:
            remaining_ledger = cap - true_spend
            reported_spend = true_spend

    overshoot = true_spend > cap
    return {
        "true_spend": true_spend,
        "reported_spend": reported_spend,
        "remaining_ledger": remaining_ledger,
        "steps": steps,
        "overshoot": overshoot,
        "overshoot_uc": max(0.0, true_spend - cap),
        "overshoot_pct": max(0.0, (true_spend - cap) / cap * 100.0),
    }

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cap", type=float, default=2000.0, help="B_0 in micro-USD")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--k", type=float, nargs="+", default=[1.0, 1.5, 2.0, 5.0, 10.0],
                   help="provider under-reporting factors (1.0 = truthful)")
    p.add_argument("--margin", type=float, default=2.0,
                   help="reservation safety margin (matches AnthropicEstimator 2.0x)")
    p.add_argument("--cost-csv", type=Path, default=None,
                   help="CSV of real per-call costs to bootstrap the cost model")
    p.add_argument("--cost-col", default="actual_cost_uc")
    p.add_argument("--reservation-col", default=None,
                   help="if set, sample (reservation, actual) PAIRS from the CSV "
                        "(correct model; preserves A1). Strongly recommended for "
                        "any real data with cost variance.")
    p.add_argument("--mean", type=float, default=928.0,
                   help="synthetic mean per-call cost (uc); LANG-001 default")
    p.add_argument("--sd", type=float, default=120.0,
                   help="synthetic sd per-call cost (uc)")
    p.add_argument("--reconcile-every", type=int, default=None,
                   help="if set, poll ground truth and correct ledger every N calls")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--output", type=Path, default=Path("a7_results.txt"))
    args = p.parse_args()

    print(f"A7 fault-injection: cap={args.cap} uc, margin={args.margin}x, "
          f"trials={args.trials} per k")
    paired = bool(args.reservation_col)
    if paired:
        if not args.cost_csv:
            raise SystemExit("--reservation-col requires --cost-csv")
        sampler = load_pair_model(args.cost_csv, args.cost_col, args.reservation_col)
    else:
        sampler = load_cost_model(args.cost_csv, args.cost_col, args.mean, args.sd)
    if args.reconcile_every:
        print(f"  reconciliation: every {args.reconcile_every} calls")
    print()

    rng = random.Random(args.seed)
    L = []
    L.append("=" * 74)
    L.append(" A7 fault injection: provider under-reporting vs cap-respecting")
    L.append("=" * 74)
    L.append(f"Cap B_0 = {args.cap:.0f} uc | margin {args.margin}x | "
             f"{args.trials} trials/k"
             + (f" | reconcile every {args.reconcile_every}" if args.reconcile_every else ""))
    L.append("")
    L.append(f"{'k':>5} {'overshoot':>12} {'95% CI':>20} "
             f"{'mean over%':>12} {'max over%':>11}")
    L.append("-" * 74)

    rows_for_interpretation = []

    for k in args.k:
        results = [run_trial(args.cap, k, args.margin, sampler, rng,
                             args.reconcile_every, paired)
                   for _ in range(args.trials)]
        n_over = sum(1 for r in results if r["overshoot"])
        lo, hi = wilson_ci(n_over, args.trials)
        over_pcts = [r["overshoot_pct"] for r in results if r["overshoot"]]
        mean_over = statistics.mean(over_pcts) if over_pcts else 0.0
        max_over = max((r["overshoot_pct"] for r in results), default=0.0)
        L.append(f"{k:>5.1f} {n_over:>5}/{args.trials:<6} "
                 f"[{lo:.3f}, {hi:.3f}]   "
                 f"{mean_over:>11.1f}% {max_over:>10.1f}%")
        rows_for_interpretation.append((k, n_over, mean_over, max_over))

    L.append("")
    L.append("Interpretation")
    L.append("-" * 74)
    txt = "\n".join(L)
    print(txt)
    args.output.write_text(txt + "\n")
    print(f"\nWrote {args.output.resolve()}")


if __name__ == "__main__":
    main()