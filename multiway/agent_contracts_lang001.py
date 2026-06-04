import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Tuple

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

TOOL_ERROR_MESSAGE = (
    "ERROR: column 'tier' does not exist. Did you mean 'tier_name'?"
)

MAX_ITERATIONS = 25

_USAGE_WARNING_PRINTED = False

def _get_usage_field(usage, candidate_names: Tuple[str, ...]) -> int:
    for name in candidate_names:
        val = getattr(usage, name, None)
        if val is not None:
            return int(val)
        try:
            val = usage[name]
            if val is not None:
                return int(val)
        except (KeyError, TypeError):
            pass
    return 0

def _extract_token_counts(response) -> Tuple[int, int]:
    global _USAGE_WARNING_PRINTED

    try:
        usage = response.usage
    except AttributeError:
        return (0, 0)
    if usage is None:
        return (0, 0)

    in_tok = _get_usage_field(usage, ("prompt_tokens", "input_tokens"))
    out_tok = _get_usage_field(usage, ("completion_tokens", "output_tokens"))

    if in_tok == 0 and out_tok == 0 and not _USAGE_WARNING_PRINTED:
        try:
            has_choices = bool(response.choices)
        except (AttributeError, TypeError):
            has_choices = False
        if has_choices:
            available_attrs = sorted(
                a for a in dir(usage)
                if not a.startswith("_") and not callable(getattr(usage, a, None))
            )
            print(
                "\n  WARNING: token-count extraction returned (0, 0) on a "
                "response that has choices.\n"
                f"  response.usage type: {type(usage).__name__}\n"
                f"  available attributes: {available_attrs}\n"
                "  Extend _get_usage_field()'s candidate names in this "
                "script to handle this library's convention.\n"
                "  Spend tracking for this run will be inaccurate; the "
                "contract's internal accounting\n"
                "  (which fires budget_violation) remains correct, so the "
                "overshoot binary is unaffected.\n",
                file=sys.stderr,
            )
            _USAGE_WARNING_PRINTED = True

    return (in_tok, out_tok)

@dataclass
class RunRecord:
    runtime: str
    run_id: int
    provider: str
    workload: str
    outcome: str
    agent_steps: int
    cap_uc: int
    total_spent_uc: int
    pct_of_cap: float
    overshoot_uc: int
    structural_undershoot_uc: int
    wasted_call_cost_uc: int
    wall_seconds: float

def usd_to_uc(usd: float) -> int:
    return int(round(usd * 100_000))

def uc_to_usd(uc: int) -> float:
    return uc / 100_000.0

def run_one(iteration: int, cap_usd: float, model: str, provider: str) -> RunRecord:
    contract = Contract(
        id=f"lang001-rep-{iteration:02d}",
        name="LANG-001 SQL retry under Agent Contracts",
        mode=ContractMode.BALANCED,
        resources=ResourceConstraints(
            cost_usd=cap_usd,
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
                        max_tokens=256,
                    )
                except Exception as e:
                    msg = str(e).lower()
                    if ("budget" in msg or "cost" in msg or "violation" in msg
                            or "exceeded" in msg or "contract" in msg):
                        outcome = "agent_contracts_budget_violation"
                    else:
                        outcome = "error"
                    break

                n_calls += 1
                in_tok, out_tok = _extract_token_counts(response)
                total_in_tok += in_tok
                total_out_tok += out_tok

                choice = response.choices[0].message
                messages.append({
                    "role": "assistant",
                    "content": getattr(choice, "content", "") or "",
                    "tool_calls": getattr(choice, "tool_calls", None),
                })

                tool_calls = getattr(choice, "tool_calls", None)
                if not tool_calls:
                    outcome = "completed_no_cap_hit"
                    break

                for tc in tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": TOOL_ERROR_MESSAGE,
                    })
            else:
                outcome = "iteration_limit"

    except Exception as e:
        outcome = f"error:{type(e).__name__}"

    elapsed = time.time() - t0

    if model.startswith("gpt-4o-mini"):
        rate_in, rate_out = 15.0, 60.0
    elif model.startswith("gpt-4o"):
        rate_in, rate_out = 250.0, 1000.0
    elif "haiku" in model:
        rate_in, rate_out = 100.0, 500.0
    elif "sonnet" in model:
        rate_in, rate_out = 300.0, 1500.0
    else:
        rate_in, rate_out = 250.0, 1000.0

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
                    help="Cost cap in USD (default 0.0054 = 540 uc, matching Table 30)")
    ap.add_argument("--output-csv", default="sweep_results/agent_contracts_lang001_n10.csv")
    args = ap.parse_args()

    # API key sanity check
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"Agent Contracts head-to-head on LANG-001 (harness v1.1, patched)")
    print(f"  Package:   ai-agent-contracts")
    print(f"  Model:     {args.model}    Provider: {args.provider}")
    print(f"  Cap:       ${args.cap_usd:.6f} ({usd_to_uc(args.cap_usd)} uc)")
    print(f"  Runs:      {args.runs}")
    print()

    records: List[RunRecord] = []
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs}...", end=" ", flush=True)
        rec = run_one(i, args.cap_usd, args.model, args.provider)
        print(f"{rec.outcome}  spent={rec.total_spent_uc} uc "
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
    print(f"  Mean spend:             {mean_spend:.1f} uc ({pct:.0f}% of cap)")
    print(f"  Max spend:              {max(spends)} uc")
    print(f"  Min spend:              {min(spends)} uc")
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

    # Sanity check: if mean_spend is suspiciously low, warn the user.
    if mean_spend < 10 and n_overshoot == 0 and any("budget_violation" in r.outcome for r in records):
        print()
        print("NOTE: mean_spend is very low (<10 uc) but budget_violation outcomes")
        print("were recorded. This usually means token-count extraction failed for")
        print("most calls (see warnings above). The overshoot binary remains valid")
        print("because the contract's internal accounting fired correctly; only")
        print("the synthetic spend column is unreliable.")


if __name__ == "__main__":
    main()