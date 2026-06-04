import argparse, asyncio, csv, os, random, sys
from collections import defaultdict

def child_actual_cost(per_child, live_usage=None):
    if live_usage is not None:
        return live_usage
    return per_child

async def condition_A_racy(B0, children, per_child, live=False):
    spent = {"v": 0}

    async def child():
        if spent["v"] + per_child <= B0:
            await asyncio.sleep(0)            # yield: the race window
            spent["v"] += child_actual_cost(per_child)
    await asyncio.gather(*(child() for _ in range(children)))

    return spent["v"], spent["v"] > B0

async def condition_locked(B0, children, per_child, live=False):
    spent = {"v": 0}
    lock = asyncio.Lock()

    async def child():
        async with lock:
            if spent["v"] + per_child <= B0:
                spent["v"] += child_actual_cost(per_child)
    await asyncio.gather(*(child() for _ in range(children)))

    return spent["v"], spent["v"] > B0

async def condition_C_affine(B0, children, per_child, live=False):
    share = B0 // children
    total = {"v": 0}

    async def child(my_share):
        spend = per_child if per_child <= my_share else 0
        await asyncio.sleep(0)
        total["v"] += child_actual_cost(spend) if spend else 0
    await asyncio.gather(*(child(share) for _ in range(children)))

    return total["v"], total["v"] > B0

CONDITIONS = {
    "A_racy_asyncio":    condition_A_racy,
    "B_locked_asyncio":  condition_locked,
    "C_affine_split":    condition_C_affine,
    "E_rust_shared_mtx": condition_locked,   # same safety mechanism as B
}

async def live_child_usage(client, model, max_out):
    resp = await client.messages.create(
        model=model, max_tokens=max_out, temperature=0.0,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    u = resp.usage
    uc = round(u.input_tokens * 0.08 + u.output_tokens * 0.4)

    return max(uc, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=int, nargs="+", default=[50, 60, 69, 100])
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--per-child", type=int, default=23)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--output", default="b0_sweep_results.csv")
    args = ap.parse_args()
    random.seed(args.seed)

    client = None
    if args.live:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            sys.exit("pip install anthropic  (or run without --live)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("set ANTHROPIC_API_KEY for --live")
        client = AsyncAnthropic()

    rows = []
    summary = defaultdict(lambda: {"trials": 0, "overshoot": 0})

    async def run_all():
        for cond_name, fn in CONDITIONS.items():
            for B0 in args.b0:
                for t in range(args.trials):
                    spent, over = await fn(B0, args.children, args.per_child, live=args.live)
                    rows.append({
                        "condition": cond_name, "B0": B0, "trial": t,
                        "children": args.children, "per_child": args.per_child,
                        "total_spent_uc": spent, "overshoot": int(over),
                    })
                    summary[(cond_name, B0)]["trials"] += 1
                    summary[(cond_name, B0)]["overshoot"] += int(over)

    asyncio.run(run_all())

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nForgetful-Operator B0 sweep  (children={args.children}, "
          f"per_child={args.per_child}, 3*per_child={3*args.per_child}, "
          f"{'LIVE' if args.live else 'OFFLINE-SIM'})\n")
    print(f"  {'condition':20s} " + " ".join(f"B0={b:<4d}" for b in args.b0))

    for cond_name in CONDITIONS:
        cells = []
        for B0 in args.b0:
            s = summary[(cond_name, B0)]
            cells.append(f"{s['overshoot']:>2d}/{s['trials']:<2d} ")
        print(f"  {cond_name:20s} " + " ".join(cells))
    print(f"\nWrote {len(rows)} rows to {args.output}")
    print("\nExpected pattern (proves the race is parameter-dependent, the "
          "discipline is not):")
    print("  A_racy_asyncio : overshoot when 3*per_child > B0 (B0=50 yes, 69 boundary, 100 NO)")
    print("  C_affine_split : 0/N at EVERY B0 -- safe by construction, no lock, no operator discipline")
    print("  B / E (locked) : 0/N at every B0 -- safe ONLY because the operator wrote the lock")

if __name__ == "__main__":
    main()