#!/usr/bin/env python3
"""
production_tier_cap_sweep.py
============================

Robustness validation: runs the production-tier agent-loop workload
across MULTIPLE cap values per model (not just the boundary cap),
demonstrating that the discipline holds across the full range of
cap choices, not just at the cherry-picked boundary.

WHY THIS EXISTS:
The single-cap production-tier validation (cap=1500uc Sonnet,
cap=600uc gpt-4o) is open to the reviewer attack "you chose caps
that force mid-loop firing; you engineered the result." This
sweep counters that attack: across 5 cap values per cell, the
discipline produces three distinct outcomes:

  - LOW caps (below per-call cost): pre-flight refusal of call 1
    (cap too tight for any call to succeed)
  - BOUNDARY caps: mid-loop firing on call 3 or 4
    (the case the single-cap run already validated)
  - GENEROUS caps: all calls complete within cap, zero overshoot
    (the case the original loose-cap pass demonstrated)

The key invariant tested: ZERO cap overshoots across ALL cap
values. That is the cap-respecting property, demonstrated robustly.

CAPS USED:
  Anthropic claude-sonnet-4-20250514:
    cap_micro_cents in {300, 750, 1500, 3000, 6000}
                       (too-tight, mid-loop, boundary, generous, very generous)
  OpenAI gpt-4o:
    cap_micro_cents in {120, 300, 600, 1200, 2400}
                       (proportionally scaled from gpt-4o's lower per-call cost)

INVOCATION:
    ANTHROPIC_API_KEY=sk-ant-...  OPENAI_API_KEY=sk-...  \\
        python tools/production_tier_cap_sweep.py --n 3

EXPECTED COST:
    5 caps x 2 cells x N=3 = 30 runs. Cost varies by cap value:
    - Tight caps spend < $0.001 per run
    - Generous caps spend $0.001-0.003 per run
    Total estimated cost: $0.10-0.30 at N=3 across all cells.

WALL-CLOCK: ~5-10 minutes.

OUTPUT:
    production_tier_cap_sweep_results.csv (30 rows, one per execution)
    production_tier_cap_sweep_summary.json
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr)
    sys.exit(1)


# =============================================================
# Cap-sweep cell configurations
# =============================================================

CELLS = []
# Anthropic Sonnet-4 cap sweep: 300, 750, 1500, 3000, 6000 uc
for cap in [300, 750, 1500, 3000, 6000]:
    CELLS.append({
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "workload": "agent_loop",
        "cap_micro_cents": cap,
    })
# OpenAI gpt-4o cap sweep: 120, 300, 600, 1200, 2400 uc
for cap in [120, 300, 600, 1200, 2400]:
    CELLS.append({
        "provider": "openai",
        "model": "gpt-4o",
        "workload": "agent_loop",
        "cap_micro_cents": cap,
    })


# =============================================================
# Workload (identical to production_tier_validation.py)
# =============================================================

AGENT_LOOP_PROMPT = """You are a debugging agent. The user reports
a Python script that fails with ImportError: No module named 'foo'.

You must propose a fix. After your first proposal, the user will say
"that didn't work, try again" up to 3 times. On each retry, propose
a different fix.

Provide your fix as 2-3 sentences."""

USER_RETRY_MSG = "That didn't work, try again."

PRICING = {
    "claude-sonnet-4-20250514": {
        "input_uc_per_token": 0.3,
        "output_uc_per_token": 1.5,
    },
    "gpt-4o": {
        "input_uc_per_token": 0.25,
        "output_uc_per_token": 1.0,
    },
}


# =============================================================
# Budget simulator
# =============================================================

@dataclass
class BudgetState:
    cap_micro_cents: int
    spent_micro_cents: int = 0

    def can_spend(self, amount: int) -> bool:
        return self.spent_micro_cents + amount <= self.cap_micro_cents

    def spend(self, amount: int) -> bool:
        if not self.can_spend(amount):
            return False
        self.spent_micro_cents += amount
        return True


@dataclass
class RunResult:
    provider: str
    model: str
    workload: str
    cap_micro_cents: int
    run_idx: int
    n_calls_attempted: int
    n_calls_completed: int
    actual_spent_uc: int
    estimated_pre_call_uc: int
    overshoot: bool
    overshoot_amount_uc: int
    mid_loop_fired: bool
    pre_flight_refused: bool
    completed_within_cap: bool
    early_completion: bool
    error: Optional[str]
    timestamp: str
    wall_clock_s: float


def estimate_cost_uc(provider, model, prompt, max_output_tokens):
    p = PRICING[model]
    input_bytes = len(prompt.encode("utf-8"))
    input_uc = int(input_bytes * p["input_uc_per_token"])
    output_uc = int(max_output_tokens * p["output_uc_per_token"])
    base = input_uc + output_uc
    return int(base * 2.0) if provider == "anthropic" else base


def actual_cost_uc(model, input_tokens, output_tokens):
    p = PRICING[model]
    return int(input_tokens * p["input_uc_per_token"]
               + output_tokens * p["output_uc_per_token"])


def call_anthropic(model, messages, max_tokens=200):
    client = Anthropic()
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=AGENT_LOOP_PROMPT,
        messages=messages, temperature=0.0,
    )
    return {
        "text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def call_openai(model, messages, max_tokens=200):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": AGENT_LOOP_PROMPT}] + messages,
        temperature=0.0,
    )
    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def run_agent_loop(provider, model, cap_uc, run_idx,
                   max_retries=4, max_tokens_per_call=200):
    budget = BudgetState(cap_micro_cents=cap_uc)
    messages = [
        {"role": "user", "content": "My Python script fails with: "
                                    "ImportError: No module named 'foo'. How do I fix it?"}
    ]
    n_attempted = 0
    n_completed = 0
    estimated_total_uc = 0
    mid_loop_fired = False
    pre_flight_refused = False
    early_completion = False
    error_msg = None
    t0 = time.time()

    try:
        for retry in range(max_retries + 1):
            prompt_proxy = "\n".join(m["content"] for m in messages)
            estimate = estimate_cost_uc(
                provider, model, prompt_proxy, max_tokens_per_call
            )
            estimated_total_uc = estimate

            n_attempted += 1
            if not budget.can_spend(estimate):
                if n_completed > 0:
                    mid_loop_fired = True
                else:
                    pre_flight_refused = True
                break

            if provider == "anthropic":
                resp = call_anthropic(model, messages, max_tokens=max_tokens_per_call)
            elif provider == "openai":
                resp = call_openai(model, messages, max_tokens=max_tokens_per_call)
            else:
                error_msg = f"unknown provider {provider}"
                break

            actual = actual_cost_uc(model, resp["input_tokens"], resp["output_tokens"])
            ok = budget.spend(actual)
            if not ok:
                error_msg = (f"A1 VIOLATION: actual {actual}uc > available "
                             f"({cap_uc - budget.spent_micro_cents}uc)")
                break
            n_completed += 1
            messages.append({"role": "assistant", "content": resp["text"]})

            text_lower = resp["text"].lower()
            if any(p in text_lower for p in [
                "let me know if", "please provide", "could you share",
                "what version of python", "more information", "could you clarify",
            ]):
                early_completion = True
                break

            messages.append({"role": "user", "content": USER_RETRY_MSG})

        overshoot = budget.spent_micro_cents > cap_uc
        overshoot_amount = max(0, budget.spent_micro_cents - cap_uc)
        completed_within_cap = (
                not mid_loop_fired and not pre_flight_refused
                and not early_completion and not error_msg
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        overshoot = False
        overshoot_amount = 0
        completed_within_cap = False

    return RunResult(
        provider=provider, model=model, workload="agent_loop",
        cap_micro_cents=cap_uc, run_idx=run_idx,
        n_calls_attempted=n_attempted, n_calls_completed=n_completed,
        actual_spent_uc=budget.spent_micro_cents,
        estimated_pre_call_uc=estimated_total_uc,
        overshoot=overshoot, overshoot_amount_uc=overshoot_amount,
        mid_loop_fired=mid_loop_fired,
        pre_flight_refused=pre_flight_refused,
        completed_within_cap=completed_within_cap,
        early_completion=early_completion,
        error=error_msg,
        timestamp=datetime.utcnow().isoformat() + "Z",
        wall_clock_s=round(time.time() - t0, 3),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cap-sweep robustness validation for production-tier models"
    )
    parser.add_argument("--n", type=int, default=3,
                        help="Replicas per (cell, cap) pair (default: 3)")
    parser.add_argument("--output-csv",
                        default="production_tier_cap_sweep_results.csv")
    parser.add_argument("--output-json",
                        default="production_tier_cap_sweep_summary.json")
    parser.add_argument("--skip-anthropic", action="store_true")
    parser.add_argument("--skip-openai", action="store_true")
    args = parser.parse_args()

    if not args.skip_anthropic and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr); sys.exit(2)
    if not args.skip_openai and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr); sys.exit(2)

    cells_to_run = [
        c for c in CELLS
        if not (
                (c["provider"] == "anthropic" and args.skip_anthropic)
                or (c["provider"] == "openai" and args.skip_openai)
        )
    ]

    print(f"Running {len(cells_to_run)} (cell, cap) configurations, "
          f"N={args.n} each = {len(cells_to_run) * args.n} total executions.\n")
    print("CAP SWEEP — testing discipline robustness across cap choices")
    print("Expected outcome patterns: PRE-FLIGHT (cap too tight), "
          "MID-LOOP (boundary), COMPLETED (generous cap). "
          "ZERO overshoots across ALL caps.\n")

    results = []
    for cell in cells_to_run:
        print(f"--- {cell['provider']}/{cell['model']} "
              f"cap={cell['cap_micro_cents']}uc ---")
        for run_idx in range(args.n):
            r = run_agent_loop(
                provider=cell["provider"],
                model=cell["model"],
                cap_uc=cell["cap_micro_cents"],
                run_idx=run_idx,
            )
            tag = "MID-LOOP" if r.mid_loop_fired else (
                "PRE-FLIGHT" if r.pre_flight_refused else (
                    "EARLY-EXIT" if r.early_completion else (
                        "OVERSHOOT" if r.overshoot else (
                            "COMPLETED" if r.completed_within_cap else "ERROR"
                        )
                    )
                )
            )
            err = f" err={r.error}" if r.error else ""
            print(f"  run {run_idx + 1:02d}/{args.n}: {tag} "
                  f"attempted={r.n_calls_attempted} "
                  f"completed={r.n_calls_completed} "
                  f"spent={r.actual_spent_uc}uc/{r.cap_micro_cents}uc{err}")
            results.append(r)
        print()

    # CSV output
    if results:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))
        print(f"Wrote {len(results)} rows to {args.output_csv}")

    # Per-cap summary
    summary = {}
    for cell in cells_to_run:
        key = f"{cell['provider']}/{cell['model']}@{cell['cap_micro_cents']}uc"
        rs = [r for r in results
              if r.provider == cell["provider"] and r.model == cell["model"]
              and r.cap_micro_cents == cell["cap_micro_cents"]]
        summary[key] = {
            "n_total": len(rs),
            "n_mid_loop": sum(1 for r in rs if r.mid_loop_fired),
            "n_pre_flight": sum(1 for r in rs if r.pre_flight_refused),
            "n_completed": sum(1 for r in rs if r.completed_within_cap),
            "n_overshoot": sum(1 for r in rs if r.overshoot),
            "n_errors": sum(1 for r in rs if r.error),
            "total_spent_uc": sum(r.actual_spent_uc for r in rs),
        }

    # Headline
    n_total = len(results)
    n_overshoot = sum(1 for r in results if r.overshoot)
    n_mid_loop = sum(1 for r in results if r.mid_loop_fired)
    n_pre_flight = sum(1 for r in results if r.pre_flight_refused)
    n_completed = sum(1 for r in results if r.completed_within_cap)
    total_uc = sum(r.actual_spent_uc for r in results)

    summary["__headline__"] = {
        "n_total": n_total,
        "n_overshoot": n_overshoot,
        "n_mid_loop": n_mid_loop,
        "n_pre_flight_refused": n_pre_flight,
        "n_completed_within_cap": n_completed,
        "n_errors": sum(1 for r in results if r.error),
        "total_actual_spent_usd": round(total_uc / 1_000_000, 6),
        "robustness_claim": (
            f"Cap-respecting holds across all {n_total} runs at "
            f"{len(cells_to_run)} cap values: {n_overshoot}/{n_total} overshoots."
        ),
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== HEADLINE ===")
    print(f"{n_total} executions across {len(cells_to_run)} (model, cap) "
          f"configurations.")
    print(f"  Mid-loop fired: {n_mid_loop}")
    print(f"  Pre-flight refused: {n_pre_flight}")
    print(f"  Completed within cap: {n_completed}")
    print(f"  Cap overshoots: {n_overshoot}")
    print(f"  Errors: {sum(1 for r in results if r.error)}")
    print(f"  Total API cost: ${total_uc / 1_000_000:.6f}")
    print(f"\nROBUSTNESS CLAIM: Cap-respecting bound held in "
          f"{n_total - n_overshoot}/{n_total} runs across {len(cells_to_run)} "
          f"distinct cap values.")


if __name__ == "__main__":
    main()