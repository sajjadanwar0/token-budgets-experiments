import argparse
import asyncio
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional
import httpx

LANG001_SYSTEM = (
    "You are a SQL agent. Use the provided sql_query tool to answer "
    "the user's question. If the first tool call does not return the "
    "expected information, try variations."
)

LANG001_USER = (
    "How many users registered in 2024 from the marketing campaign "
    "table? The table schema is: users(id, email, signup_source, "
    "signup_date). Use the sql_query tool."
)


@dataclass
class TrialResult:
    trial_id: int
    cap_uc: int
    total_spent_uc: int
    overshoot: bool
    children_admitted: int
    children_completed: int
    elapsed_s: float
    error: Optional[str] = None


class LockedBudget:
    def __init__(self, cap_uc: int):
        self.cap_uc = cap_uc
        self.spent_uc = 0
        self.lock = asyncio.Lock()

    async def try_reserve(self, estimate_uc: int) -> bool:
        async with self.lock:
            if self.spent_uc + estimate_uc > self.cap_uc:
                return False
            self.spent_uc += estimate_uc
            return True

    async def refund(self, refund_uc: int) -> None:
        async with self.lock:
            self.spent_uc -= refund_uc

    async def forfeit_excess(self, actual_uc: int,
                             reserved_uc: int) -> None:
        async with self.lock:
            if actual_uc > reserved_uc:
                self.spent_uc += (actual_uc - reserved_uc)


def estimate_uc_byte_length(prompt: str, max_output_tokens: int,
                            rate_in_per_mtok: float,
                            rate_out_per_mtok: float,
                            margin: float = 0.5) -> int:
    input_tokens_est = int(margin * len(prompt))
    in_uc = int(input_tokens_est * rate_in_per_mtok / 10)
    out_uc = int(max_output_tokens * rate_out_per_mtok / 10)
    return in_uc + out_uc


async def call_anthropic(client: httpx.AsyncClient, api_key: str,
                         prompt_system: str, prompt_user: str,
                         max_output_tokens: int) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "system": prompt_system,
        "messages": [{"role": "user", "content": prompt_user}],
    }

    r = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
    r.raise_for_status()

    return r.json()


async def child(child_id: int, budget: LockedBudget,
                api_key: str, client: httpx.AsyncClient,
                rate_in: float, rate_out: float,
                max_output_tokens: int, margin: float) -> dict:
    estimate = estimate_uc_byte_length(
        prompt=LANG001_SYSTEM + LANG001_USER,
        max_output_tokens=max_output_tokens,
        rate_in_per_mtok=rate_in,
        rate_out_per_mtok=rate_out,
        margin=margin,
    )

    if not await budget.try_reserve(estimate):
        return {"child_id": child_id, "admitted": False,
                "actual_uc": 0}

    response = await call_anthropic(
        client, api_key, LANG001_SYSTEM, LANG001_USER,
        max_output_tokens=max_output_tokens,
    )

    usage = response.get("usage", {})
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    actual_uc = int(in_tokens * rate_in / 10) + int(out_tokens * rate_out / 10)

    if actual_uc < estimate:
        await budget.refund(estimate - actual_uc)
    elif actual_uc > estimate:
        await budget.forfeit_excess(actual_uc, estimate)

    return {
        "child_id": child_id,
        "admitted": True,
        "actual_uc": actual_uc,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
    }

async def run_trial(trial_id: int, cap_uc: int,
                    api_key: str, client: httpx.AsyncClient,
                    rate_in: float, rate_out: float,
                    max_output_tokens: int, margin: float) -> TrialResult:
    t0 = time.monotonic()
    budget = LockedBudget(cap_uc=cap_uc)
    try:
        results = await asyncio.gather(
            child(0, budget, api_key, client, rate_in, rate_out,
                  max_output_tokens, margin),
            child(1, budget, api_key, client, rate_in, rate_out,
                  max_output_tokens, margin),
            child(2, budget, api_key, client, rate_in, rate_out,
                  max_output_tokens, margin),
        )
        admitted = sum(1 for r in results if r["admitted"])
        completed = sum(1 for r in results if r.get("actual_uc", 0) > 0)
        return TrialResult(
            trial_id=trial_id,
            cap_uc=cap_uc,
            total_spent_uc=budget.spent_uc,
            overshoot=(budget.spent_uc > cap_uc),
            children_admitted=admitted,
            children_completed=completed,
            elapsed_s=time.monotonic() - t0,
        )
    except Exception as e:
        return TrialResult(
            trial_id=trial_id,
            cap_uc=cap_uc,
            total_spent_uc=budget.spent_uc,
            overshoot=False,
            children_admitted=0,
            children_completed=0,
            elapsed_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
        )

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--output", type=str,
                    default="results/python_locked_anthropic.csv")
    ap.add_argument("--rate-in", type=float, default=1.0)
    ap.add_argument("--rate-out", type=float, default=5.0)
    # NEW knobs (matching python_racy):
    ap.add_argument("--max-output-tokens", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    estimate = estimate_uc_byte_length(
        LANG001_SYSTEM + LANG001_USER,
        args.max_output_tokens, args.rate_in, args.rate_out, args.margin,
    )

    print(f"Running python_locked: N={args.n}, cap={args.cap} uc")
    print(f"  estimate per child = {estimate} uc")

    results: List[TrialResult] = []

    async with httpx.AsyncClient() as client:
        for i in range(args.n):
            r = await run_trial(i, args.cap, api_key, client,
                                args.rate_in, args.rate_out,
                                args.max_output_tokens, args.margin)
            results.append(r)
            print(f"  trial {i}: spent={r.total_spent_uc} uc, "
                  f"overshoot={r.overshoot}, "
                  f"admitted={r.children_admitted}/3, "
                  f"elapsed={r.elapsed_s:.1f}s"
                  + (f", ERR: {r.error}" if r.error else ""))

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "cap_uc", "total_spent_uc", "overshoot",
                    "children_admitted", "children_completed",
                    "elapsed_s", "error"])
        for r in results:
            w.writerow([r.trial_id, r.cap_uc, r.total_spent_uc,
                        int(r.overshoot), r.children_admitted,
                        r.children_completed, f"{r.elapsed_s:.3f}",
                        r.error or ""])

    overshoots = sum(1 for r in results if r.overshoot)
    print(f"\nSUMMARY: {overshoots}/{args.n} overshoots")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())