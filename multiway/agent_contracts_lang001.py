#!/usr/bin/env python3
"""
agent_contracts_lang001.py — Ye & Tan Agent Contracts head-to-head on LANG-001.

WHAT THIS DOES
==============
Reproduces the LANG-001 SQL-retry workload under Agent Contracts' resource
budget enforcement, parallel to the six runtimes already in Table 30 of
the paper. Output CSV uses the same schema as multiway_compare.py so the
result drops directly into Table 30 as a seventh row.

Tested against ai-agent-contracts v0.3.1 (April 2026). API per
flyersworder/agent-contracts README (Q. Ye, J. Tan, arXiv:2601.08815).

INSTALLATION
============
This script assumes you've installed agent-contracts in your existing
multiway venv. If not:

  cd /path/to/token-budget-experiments/multiway
  uv add ai-agent-contracts
  uv sync
  uv run python -c "import agent_contracts; print(agent_contracts.__version__)"

USAGE
=====
  export OPENAI_API_KEY=sk-...
  uv run python agent_contracts_lang001.py \
      --provider openai --model gpt-4o --runs 10 \
      --cap-usd 0.0054 \
      --output-csv sweep_results/agent_contracts_lang001_n10.csv

(cost_usd=0.0054 == 540 micro-cents, matching Table 30's cap.)

Cost: ~$0.50 for N=10 on gpt-4o. Run on gpt-4o-mini or anthropic
claude-haiku-4-5 for cheaper experimentation.

WHAT THE OUTCOME WILL TELL US
=============================
The paper expects Agent Contracts to behave like a post-call observer
(its enforcement layer is runtime cost monitoring, not compile-time
typing). If that's right, we expect:

  - Mean overshoot ≈ 1× the cap-crossing call cost (similar to LiteLLM
    and AgentGuard rows in Table 30, ~181% of cap)
  - 10/10 overshoot (always admits the threshold-crossing call)

If instead it does pre-flight reservation (like Token Budgets), we
expect 0/10 overshoot. Either result is publishable: the paper text
in §6.2.0.4 currently characterizes their work as "complementary" with
runtime hard enforcement; the empirical measurement either confirms
or refines that characterization.
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List

try:
    from agent_contracts import (
        Contract,
        ContractedLLM,
        ResourceConstraints,
        ContractMode,
    )
except ImportError:
    sys.exit(
        "ERROR: agent_contracts not importable.\n"
        "Install with: uv add ai-agent-contracts (or: pip install ai-agent-contracts)\n"
        "Importable name is 'agent_contracts' (no hyphen)."
    )


# === LANG-001 workload (same as the other runners in Table 30) ===

SQL_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Execute a SQL query against the customer database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"},
            },
            "required": ["query"],
        },
    },
}

USER_PROMPT = (
    "Use the execute_sql tool to find all customers in the 'enterprise' "
    "tier whose contracts expire in Q1 2026. If the query fails, inspect "
    "the error message and retry with a corrected query."
)

# Simulated tool error: forces the LANG-001 retry-loop pattern.
TOOL_ERROR_MESSAGE = (
    "ERROR: column 'tier' does not exist. Did you mean 'tier_name'?"
)

MAX_ITERATIONS = 25  # outer safety bound; should never bind


# === Output schema ===

@dataclass
class RunRecord:
    """Schema matches sweep_results/gpt4o_lang001_n10_full.csv so this CSV
    drops into Table 30 as an additional row."""
    runtime: str
    run_id: int
    provider: str
    workload: str
    outcome: str            # 'completed_no_cap_hit', 'agent_contracts_budget_violation',
                            # 'agent_contracts_hard_stop', 'iteration_limit', 'error'
    agent_steps: int
    cap_uc: int
    total_spent_uc: int
    pct_of_cap: float
    overshoot_uc: int
    structural_undershoot_uc: int
    wasted_call_cost_uc: int
    wall_seconds: float


def usd_to_uc(usd: float) -> int:
    """Convert dollars to micro-cents (1 µ¢ = $10^-5)."""
    return int(round(usd * 100_000))


def uc_to_usd(uc: int) -> float:
    return uc / 100_000.0


def run_one(iteration: int, cap_usd: float, model: str, provider: str) -> RunRecord:
    """Single LANG-001 replica under Agent Contracts."""

    contract = Contract(
        id=f"lang001-rep-{iteration:02d}",
        name="LANG-001 SQL retry under Agent Contracts",
        # BALANCED mode keeps the comparison most like the other runtimes
        # (no aggressive cost-shaving from ECONOMICAL or extra-token bias
        # from URGENT).
        mode=ContractMode.BALANCED,
        resources=ResourceConstraints(
            cost_usd=cap_usd,
            # Leaving tokens / api_calls unset so cost is the sole binding
            # constraint, matching Table 30's "cap_uc" semantics.
        ),
    )

    cap_uc = usd_to_uc(cap_usd)
    n_calls = 0
    total_in_tok = 0
    total_out_tok = 0
    outcome = "completed_no_cap_hit"

    messages = [{"role": "user", "content": USER_PROMPT}]

    t0 = time.time()
    try:
        with ContractedLLM(contract) as llm:
            for step in range(MAX_ITERATIONS):
                try:
                    response = llm.completion(
                        model=model,
                        messages=messages,
                        tools=[SQL_TOOL_SPEC],
                        # Modest output cap to keep per-call cost predictable.
                        max_tokens=256,
                    )
                except Exception as e:
                    # Agent Contracts raises on budget violation. Catch and
                    # categorise.
                    msg = str(e).lower()
                    if ("budget" in msg or "cost" in msg or "violation" in msg
                            or "exceeded" in msg or "contract" in msg):
                        outcome = "agent_contracts_budget_violation"
                    else:
                        outcome = "error"
                    break

                n_calls += 1

                # Extract usage (LiteLLM-compatible response shape).
                try:
                    usage = response.usage
                    total_in_tok += getattr(usage, "prompt_tokens", 0) or 0
                    total_out_tok += getattr(usage, "completion_tokens", 0) or 0
                except AttributeError:
                    pass

                # LiteLLM response shape: response.choices[0].message
                choice = response.choices[0].message
                # Append assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": getattr(choice, "content", "") or "",
                    "tool_calls": getattr(choice, "tool_calls", None),
                })

                tool_calls = getattr(choice, "tool_calls", None)
                if not tool_calls:
                    # Agent gave up or terminated naturally.
                    outcome = "completed_no_cap_hit"
                    break

                # Execute tool (simulated SQL error to force the retry pattern).
                for tc in tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": TOOL_ERROR_MESSAGE,
                    })
            else:
                outcome = "iteration_limit"

    except Exception as e:
        # Top-level failure (auth, network, etc.)
        outcome = f"error:{type(e).__name__}"

    elapsed = time.time() - t0

    # Compute synthetic cost in micro-cents using the same rates as
    # multiway_compare.py so the row is dimensionally comparable to
    # Table 30. (Agent Contracts tracks cost internally too, but we
    # recompute here for schema parity.)
    if model.startswith("gpt-4o-mini"):
        rate_in, rate_out = 15.0, 60.0     # µ¢/Mtok at $0.15 / $0.60
    elif model.startswith("gpt-4o"):
        rate_in, rate_out = 250.0, 1000.0  # µ¢/Mtok at $2.50 / $10.00
    elif "haiku" in model:
        rate_in, rate_out = 100.0, 500.0   # µ¢/Mtok at $1.00 / $5.00
    elif "sonnet" in model:
        rate_in, rate_out = 300.0, 1500.0  # µ¢/Mtok at $3.00 / $15.00
    else:
        rate_in, rate_out = 250.0, 1000.0  # fallback to gpt-4o rates

    cost_uc = int(round(total_in_tok * rate_in / 1_000_000.0
                        + total_out_tok * rate_out / 1_000_000.0))
    overshoot = max(0, cost_uc - cap_uc)
    pct = 100.0 * cost_uc / cap_uc if cap_uc > 0 else 0.0

    return RunRecord(
        runtime="agent_contracts",
        run_id=iteration + 1,
        provider=provider,
        workload="lang001",
        outcome=outcome,
        agent_steps=n_calls,
        cap_uc=cap_uc,
        total_spent_uc=cost_uc,
        pct_of_cap=round(pct, 2),
        overshoot_uc=overshoot,
        structural_undershoot_uc=0,
        wasted_call_cost_uc=0,
        wall_seconds=round(elapsed, 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default="gpt-4o",
                    help="Model name passed to LiteLLM (e.g. gpt-4o, gpt-4o-mini, "
                         "anthropic/claude-haiku-4-5)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--cap-usd", type=float, default=0.0054,
                    help="Cost cap in USD (default 0.0054 = 540 µ¢, matching Table 30)")
    ap.add_argument("--output-csv", default="sweep_results/agent_contracts_lang001_n10.csv")
    args = ap.parse_args()

    # API key sanity check
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"Agent Contracts head-to-head on LANG-001")
    print(f"  Package:   ai-agent-contracts")
    print(f"  Model:     {args.model}    Provider: {args.provider}")
    print(f"  Cap:       ${args.cap_usd:.6f} ({usd_to_uc(args.cap_usd)} µ¢)")
    print(f"  Runs:      {args.runs}")
    print()

    records: List[RunRecord] = []
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs}...", end=" ", flush=True)
        rec = run_one(i, args.cap_usd, args.model, args.provider)
        print(f"{rec.outcome}  spent={rec.total_spent_uc} µ¢ "
              f"({rec.pct_of_cap:.0f}% of cap)  "
              f"calls={rec.agent_steps}  overshoot={rec.overshoot_uc}  "
              f"[{rec.wall_seconds:.1f}s]")
        records.append(rec)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(records[0])])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"\nWrote {args.output_csv}")

    # Summary
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    spends = [r.total_spent_uc for r in records]
    overshoots = [r.overshoot_uc for r in records]
    n_overshoot = sum(1 for o in overshoots if o > 0)
    mean_spend = sum(spends) / len(spends)
    cap = records[0].cap_uc
    pct = 100.0 * mean_spend / cap
    print(f"  Runtime:                agent_contracts")
    print(f"  Mean spend:             {mean_spend:.1f} µ¢ ({pct:.0f}% of cap)")
    print(f"  Max spend:              {max(spends)} µ¢")
    print(f"  Min spend:              {min(spends)} µ¢")
    print(f"  Overshoot count:        {n_overshoot}/{len(records)}")
    print(f"  Outcomes:               "
          f"{dict((o, sum(1 for r in records if r.outcome == o)) for o in set(r.outcome for r in records))}")
    print()
    print("INTERPRETATION")
    print("-" * 65)
    if n_overshoot == len(records):
        print("  All runs overshoot the cap. Agent Contracts is behaving like a")
        print("  post-call observer (similar to LiteLLM/AgentGuard in Table 30):")
        print("  the threshold-crossing call is admitted before enforcement fires.")
    elif n_overshoot == 0:
        print("  No runs overshoot. Agent Contracts is behaving like a pre-flight")
        print("  reservation system (similar to Token Budgets in Table 30): it")
        print("  refuses cap-violating calls before they execute.")
    else:
        print(f"  Mixed: {n_overshoot}/{len(records)} runs overshoot. Worth inspecting")
        print("  the runs.csv to see if a particular workload trajectory bypasses")
        print("  the enforcement layer.")


if __name__ == "__main__":
    main()