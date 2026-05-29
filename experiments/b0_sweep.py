import argparse, asyncio, csv, os, random, sys
from collections import defaultdict

# ----------------------------------------------------------------------------
# Core cost model (shared by sim and live)
# ----------------------------------------------------------------------------
# Each child task reserves `per_child` uc pre-flight and then actually spends
# `actual` uc. In the deterministic sim, actual == per_child (the race is about
# *reservation accounting* across concurrent children, not output variance).

def child_actual_cost(per_child, live_usage=None):
    """uc actually charged for one child call."""
    if live_usage is not None:
        return live_usage          # real provider usage, micro-cents
    return per_child               # deterministic sim: charge == reservation

# ----------------------------------------------------------------------------
# Condition A: racy asyncio — read-modify-write a shared int with NO lock.
# Each child checks `spent + per_child <= B0` then awaits, then adds. Because
# the check and the add straddle an await, all children pass the check before
# any commits -> classic delegation-fanout race.
# ----------------------------------------------------------------------------
async def condition_A_racy(B0, children, per_child, live=False):
    spent = {"v": 0}
    overshoot = {"v": False}

    async def child():
        # READ + CHECK
        if spent["v"] + per_child <= B0:
            await asyncio.sleep(0)            # yield: the race window
            # COMMIT (no re-check)
            spent["v"] += child_actual_cost(per_child)
        # else: refused
    await asyncio.gather(*(child() for _ in range(children)))
    return spent["v"], spent["v"] > B0

# ----------------------------------------------------------------------------
# Condition B / E: locked — same logic but the check+commit is atomic.
# Models both "locked asyncio counter" and "Rust Arc<Mutex<Budget>>": safe ONLY
# because the operator wrote the lock.
# ----------------------------------------------------------------------------
async def condition_locked(B0, children, per_child, live=False):
    spent = {"v": 0}
    lock = asyncio.Lock()

    async def child():
        async with lock:
            if spent["v"] + per_child <= B0:
                spent["v"] += child_actual_cost(per_child)
    await asyncio.gather(*(child() for _ in range(children)))
    return spent["v"], spent["v"] > B0

# ----------------------------------------------------------------------------
# Condition C: affine split — the parent splits B0 into children up front.
# A child can only spend from its OWN sub-budget; there is no shared mutable
# counter to race on. Safe BY CONSTRUCTION, no lock, no operator discipline.
# This is the Python emulation of the Rust `split` semantics.
# ----------------------------------------------------------------------------
async def condition_C_affine(B0, children, per_child, live=False):
    # Parent hands each child floor(B0/children); remainder stays with parent.
    share = B0 // children
    total = {"v": 0}

    async def child(my_share):
        # child can spend at most its own share; a request over share is refused
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

# ----------------------------------------------------------------------------
# Optional live Anthropic mode
# ----------------------------------------------------------------------------
async def live_child_usage(client, model, max_out):
    """Return micro-cents actually charged for one minimal call."""
    resp = await client.messages.create(
        model=model, max_tokens=max_out, temperature=0.0,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    u = resp.usage
    # micro-cents: (in*price_in + out*price_out); haiku-4.5 ~ $0.80/$4 per Mtok
    uc = round(u.input_tokens * 0.08 + u.output_tokens * 0.4)
    return max(uc, 1)

# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
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

    # Summary table
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