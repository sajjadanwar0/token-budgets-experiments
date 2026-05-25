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


class RacyBudget:
    def __init__(self, cap_uc: int):
        self.cap_uc = cap_uc
        self.spent_uc = 0

    async def can_admit(self, estimate_uc: int) -> bool:
        return self.spent_uc + estimate_uc <= self.cap_uc

    async def record_spend(self, actual_uc: int) -> None:
        self.spent_uc += actual_uc


def estimate_uc_byte_length(prompt: str, max_output_tokens: int,
                            rate_in_per_mtok: float,
                            rate_out_per_mtok: float,
                            margin: float = 0.5) -> int:
    """Tighter estimator (margin=0.5) -- byte-length is over-counting
    English tokens by ~4x already; 0.5x gives an estimate ~2x actual,
    which is still A1-sound but tight enough that 3*actual can exceed
    a cap that admits estimate<cap individually.
    """
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
    r = await client.post("https://api.anthropic.com/v1/messages",
                          headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


async def child(child_id: int, budget: RacyBudget,
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

    if not await budget.can_admit(estimate):
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

    await budget.record_spend(actual_uc)

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
    budget = RacyBudget(cap_uc=cap_uc)
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
    # CHANGED: cap default 100 -> 60
    ap.add_argument("--cap", type=int, default=60,
                    help="parent budget in micro-cents")
    ap.add_argument("--output", type=str,
                    default="results/python_racy_anthropic.csv")
    ap.add_argument("--rate-in", type=float, default=1.0)
    ap.add_argument("--rate-out", type=float, default=5.0)
    # NEW knobs:
    ap.add_argument("--max-output-tokens", type=int, default=30,
                    help="per-call max_tokens (CHANGED from 200 default)")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="byte-length-to-token margin (CHANGED from 2.0)")
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
    print(f"Running python_racy: N={args.n}, cap={args.cap} uc")
    print(f"  estimate per child = {estimate} uc "
          f"(margin={args.margin}, max_out={args.max_output_tokens})")
    print(f"  pre-flight check: estimate < cap? "
          f"{estimate < args.cap}  ({estimate} < {args.cap})")

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
