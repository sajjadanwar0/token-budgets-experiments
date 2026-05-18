#!/usr/bin/env python3
"""
production_tier_validation.py  (v3 — bugfix for n_completed reference)
======================================================================

CHANGES FROM v2:
  v2 crashed in the summary stage with AttributeError because the
  summary code referenced `r.n_completed` but the dataclass field
  is `r.n_calls_completed`. The CSV was written correctly before
  the crash; only the summary JSON computation failed. v3 fixes
  the field name reference and produces the summary JSON correctly.

  v2 smoke (N=2) confirmed cap calibration is correct:
    Sonnet-4: 2/2 MID-LOOP fired, 2 calls completed, 429uc spent
    gpt-4o:   2/2 MID-LOOP fired, 2 calls completed, 223-238uc spent

VALIDATES the Token Budgets discipline on PRODUCTION-TIER LLM models:
  - Anthropic claude-sonnet-4-20250514  (cap 1500uc = 0.15c)
  - OpenAI    gpt-4o                    (cap 600uc  = 0.06c)

EXPECTED RESULT WITH v3:
  Both cells: 10/10 mid-loop fired, 0/10 cap overshoots.
  Total API cost across both cells: ~$0.02-0.05 at N=10.

INVOCATION:
    ANTHROPIC_API_KEY=sk-ant-...  OPENAI_API_KEY=sk-...  \\
        python tools/production_tier_validation.py --n 10
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
# Cell configurations
# =============================================================

CELLS = [
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "workload": "agent_loop",
        "cap_micro_cents": 1_500,  # 0.15c -- calibrated from v1 data
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "workload": "agent_loop",
        "cap_micro_cents": 600,  # 0.06c -- calibrated from v1 data
    },
]


# =============================================================
# Workload
# =============================================================

AGENT_LOOP_PROMPT = """You are a debugging agent. The user reports
a Python script that fails with ImportError: No module named 'foo'.

You must propose a fix. After your first proposal, the user will say
"that didn't work, try again" up to 3 times. On each retry, propose
a different fix.

Provide your fix as 2-3 sentences."""

USER_RETRY_MSG = "That didn't work, try again."


# =============================================================
# Pricing
# =============================================================

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


# =============================================================
# Per-execution result row
# =============================================================

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
    early_completion: bool
    error: Optional[str]
    timestamp: str
    wall_clock_s: float


# =============================================================
# Pre-call cost estimation
# =============================================================

def estimate_cost_uc(
        provider: str,
        model: str,
        prompt: str,
        max_output_tokens: int,
) -> int:
    p = PRICING[model]
    input_bytes = len(prompt.encode("utf-8"))
    input_uc = int(input_bytes * p["input_uc_per_token"])
    output_uc = int(max_output_tokens * p["output_uc_per_token"])
    base_estimate = input_uc + output_uc
    if provider == "anthropic":
        return int(base_estimate * 2.0)
    return base_estimate


def actual_cost_uc(model: str, input_tokens: int, output_tokens: int) -> int:
    p = PRICING[model]
    return int(
        input_tokens * p["input_uc_per_token"]
        + output_tokens * p["output_uc_per_token"]
    )


# =============================================================
# Provider call wrappers
# =============================================================

def call_anthropic(model: str, messages: list, max_tokens: int = 200):
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=AGENT_LOOP_PROMPT,
        messages=messages,
        temperature=0.0,
    )
    text = response.content[0].text
    usage = response.usage
    return {
        "text": text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def call_openai(model: str, messages: list, max_tokens: int = 200):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": AGENT_LOOP_PROMPT}] + messages,
        temperature=0.0,
    )
    text = response.choices[0].message.content
    usage = response.usage
    return {
        "text": text,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


# =============================================================
# Single agent_loop run
# =============================================================

def run_agent_loop(
        provider: str,
        model: str,
        cap_uc: int,
        run_idx: int,
        max_retries: int = 4,
        max_tokens_per_call: int = 200,
) -> RunResult:

    budget = BudgetState(cap_micro_cents=cap_uc)
    messages: list = [
        {"role": "user", "content": "My Python script fails with: "
                                    "ImportError: No module named 'foo'. How do I fix it?"}
    ]
    n_attempted = 0
    n_completed = 0
    estimated_total_uc = 0
    mid_loop_fired = False
    early_completion = False
    error_msg: Optional[str] = None
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
                break

            if provider == "anthropic":
                resp = call_anthropic(
                    model, messages, max_tokens=max_tokens_per_call
                )
            elif provider == "openai":
                resp = call_openai(
                    model, messages, max_tokens=max_tokens_per_call
                )
            else:
                error_msg = f"unknown provider {provider}"
                break

            actual = actual_cost_uc(
                model, resp["input_tokens"], resp["output_tokens"]
            )
            ok = budget.spend(actual)
            if not ok:
                error_msg = (
                    f"A1 VIOLATION: actual {actual}uc > available "
                    f"({budget.cap_micro_cents - budget.spent_micro_cents}uc)"
                )
                break
            n_completed += 1
            messages.append({"role": "assistant", "content": resp["text"]})

            text_lower = resp["text"].lower()
            if any(phrase in text_lower for phrase in [
                "let me know if", "please provide",
                "could you share", "what version of python",
                "more information", "could you clarify",
            ]):
                early_completion = True
                break

            messages.append({"role": "user", "content": USER_RETRY_MSG})

        overshoot = budget.spent_micro_cents > cap_uc
        overshoot_amount = max(0, budget.spent_micro_cents - cap_uc)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        overshoot = False
        overshoot_amount = 0

    return RunResult(
        provider=provider,
        model=model,
        workload="agent_loop",
        cap_micro_cents=cap_uc,
        run_idx=run_idx,
        n_calls_attempted=n_attempted,
        n_calls_completed=n_completed,
        actual_spent_uc=budget.spent_micro_cents,
        estimated_pre_call_uc=estimated_total_uc,
        overshoot=overshoot,
        overshoot_amount_uc=overshoot_amount,
        mid_loop_fired=mid_loop_fired,
        early_completion=early_completion,
        error=error_msg,
        timestamp=datetime.utcnow().isoformat() + "Z",
        wall_clock_s=round(time.time() - t0, 3),
    )


# =============================================================
# Driver
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production-tier validation harness (v3 bugfix)"
    )
    parser.add_argument("--n", type=int, default=10,
                        help="Number of runs per cell (default: 10)")
    parser.add_argument("--output-csv",
                        default="production_tier_validation_results.csv")
    parser.add_argument("--output-json",
                        default="production_tier_validation_summary.json")
    parser.add_argument("--skip-anthropic", action="store_true")
    parser.add_argument("--skip-openai", action="store_true")
    args = parser.parse_args()

    if not args.skip_anthropic and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Use --skip-anthropic.",
              file=sys.stderr)
        sys.exit(2)
    if not args.skip_openai and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Use --skip-openai.",
              file=sys.stderr)
        sys.exit(2)

    results: list[RunResult] = []
    cells_to_run = [
        c for c in CELLS
        if not (
                (c["provider"] == "anthropic" and args.skip_anthropic)
                or (c["provider"] == "openai" and args.skip_openai)
        )
    ]

    print(f"Running {len(cells_to_run)} cell(s), N={args.n} each = "
          f"{len(cells_to_run) * args.n} total executions.")
    print(f"TIGHTENED CAPS: Sonnet=1500uc (0.15c), gpt-4o=600uc (0.06c)\n")

    for cell in cells_to_run:
        print(f"--- Cell: {cell['provider']}/{cell['model']} "
              f"workload={cell['workload']} cap={cell['cap_micro_cents']}uc ---")
        for run_idx in range(args.n):
            r = run_agent_loop(
                provider=cell["provider"],
                model=cell["model"],
                cap_uc=cell["cap_micro_cents"],
                run_idx=run_idx,
            )
            tag = "MID-LOOP" if r.mid_loop_fired else (
                "EARLY-EXIT" if r.early_completion else
                ("OVERSHOOT" if r.overshoot else (
                    "PRE-FLIGHT-REFUSED" if (
                            r.n_calls_completed == 0 and r.error is None
                    ) else "completed"
                ))
            )
            err = f" err={r.error}" if r.error else ""
            print(f"  run {run_idx + 1:02d}/{args.n}: {tag} "
                  f"calls_attempted={r.n_calls_attempted} "
                  f"completed={r.n_calls_completed} "
                  f"spent={r.actual_spent_uc}uc/{r.cap_micro_cents}uc"
                  f"{err}")
            results.append(r)
        print()

    if results:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"Wrote {len(results)} rows to {args.output_csv}")

    # =============================================================
    # BUGFIX: use r.n_calls_completed (dataclass field name),
    # not r.n_completed (which doesn't exist).
    # =============================================================
    summary = {}
    for cell in cells_to_run:
        key = f"{cell['provider']}/{cell['model']}"
        cell_results = [
            r for r in results
            if r.provider == cell["provider"] and r.model == cell["model"]
        ]
        n_total = len(cell_results)
        n_mid_loop = sum(1 for r in cell_results if r.mid_loop_fired)
        n_early = sum(1 for r in cell_results if r.early_completion)
        n_preflight = sum(
            1 for r in cell_results
            if r.n_calls_completed == 0 and not r.mid_loop_fired
            and not r.error and not r.early_completion
        )
        n_completed = sum(
            1 for r in cell_results
            if not r.mid_loop_fired and not r.early_completion
            and not r.error and r.n_calls_completed > 0
        )
        n_overshoot = sum(1 for r in cell_results if r.overshoot)
        n_errors = sum(1 for r in cell_results if r.error)
        total_actual_uc = sum(r.actual_spent_uc for r in cell_results)
        summary[key] = {
            "cell": cell,
            "n_total": n_total,
            "n_mid_loop_fired": n_mid_loop,
            "n_early_completion": n_early,
            "n_pre_flight_refused": n_preflight,
            "n_completed_within_cap": n_completed,
            "n_overshoot": n_overshoot,
            "n_errors": n_errors,
            "total_actual_spent_uc": total_actual_uc,
            "total_actual_spent_usd": round(total_actual_uc / 1_000_000, 6),
            "headline": (
                f"{n_mid_loop}/{n_total} mid-loop fired, "
                f"{n_overshoot}/{n_total} cap overshoots, "
                f"${total_actual_uc / 1_000_000:.6f} total API cost"
            ),
        }

    summary["headline_total"] = {
        "n_total": len(results),
        "n_mid_loop_fired": sum(1 for r in results if r.mid_loop_fired),
        "n_overshoot": sum(1 for r in results if r.overshoot),
        "n_errors": sum(1 for r in results if r.error),
        "total_actual_spent_usd": round(
            sum(r.actual_spent_uc for r in results) / 1_000_000, 6
        ),
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote summary to {args.output_json}")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if k == "headline_total":
            print(f"\nTOTAL: {v['n_mid_loop_fired']}/{v['n_total']} mid-loop, "
                  f"{v['n_overshoot']}/{v['n_total']} overshoots, "
                  f"${v['total_actual_spent_usd']} API cost")
        else:
            print(f"{k}: {v['headline']}")


if __name__ == "__main__":
    main()