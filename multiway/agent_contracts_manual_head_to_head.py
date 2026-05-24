#!/usr/bin/env python3
"""agent_contracts_manual_head_to_head.py — fallback if ContractedLLM is broken.

Uses Agent Contracts' Contract and ResourceConstraints types as the
contract SPECIFICATION (so AC's data model defines the cap), but
enforces the cap manually via litellm — bypassing the ContractedLLM
context manager that's failing on v0.3.1 with 'NoneType is not callable'.

This is methodologically defensible for a head-to-head: both Token
Budgets and Agent Contracts are pre-flight cost-check disciplines.
Token Budgets enforces via an affine-typed Budget value; Agent
Contracts enforces via a Contract instance that wraps a ContractedLLM.
For comparable measurement, what matters is: same cap, same workload,
same model, same temperature, same N — both run pre-flight cost
checks and reject if cap would be exceeded. This script does exactly
that, using AC's spec types so the contract definition is identical
to what ContractedLLM would consume.

If you get ContractedLLM working via ac_diagnostic.py, use
agent_contracts_head_to_head.py instead — that's the more direct
comparison. This script is for the case where v0.3.1's integration
wrapper is broken on your machine.

Run:
  source ~/.zshrc
  source .ac_venv/bin/activate
  pip install ai-agent-contracts==0.3.1 litellm
  python3 multiway/agent_contracts_manual_head_to_head.py
"""

from __future__ import annotations
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from agent_contracts import Contract, ResourceConstraints, ContractMode
    AC_AVAILABLE = True
except ImportError:
    print("WARNING: ai-agent-contracts not installed; using equivalent local spec")
    AC_AVAILABLE = False

try:
    import litellm
except ImportError:
    print("ERROR: litellm not installed. Run: pip install litellm")
    sys.exit(1)

LANG_001_PROMPT = (
    "You are a customer service agent. A customer just wrote: "
    '"I want to cancel my order but I also need to know if I can keep '
    'the free gift that came with it." '
    "Before responding, identify exactly what clarifications you need "
    "from the customer to handle this correctly. Then ask the most "
    "important clarification question."
)

MODEL_LITELLM = "claude-sonnet-4-5-20250929"   # try bare name first
CAP_USD = 540 / 1_000_000   # 540 uc = $0.000540 (matches TB convention)
N_TRIALS = 30
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 200
SONNET_INPUT_USD_PER_TOK = 3.0  / 1_000_000
SONNET_OUTPUT_USD_PER_TOK = 15.0 / 1_000_000

OUT_DIR = Path(__file__).resolve().parent / "sweep_results"
OUT_PATH = OUT_DIR / "agent_contracts_lang001_cap540_n30_anthropic.csv"


@dataclass
class TrialResult:
    trial_id: int
    workload: str
    framework: str
    cap_usd: float
    cap_uc_equiv: int
    n_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_cost_uc_equiv: int
    overshoot: int
    refused_pre_flight: int
    refused_reason: str
    response_truncated: str
    wall_time_s: float


def estimate_input_cost_usd(prompt: str) -> float:
    """Pre-flight input-cost estimate: prompt bytes / 4 char-per-tok approx.
    Conservative for Anthropic (real tok/byte ~0.5-0.85 on English)."""
    approx_tokens = len(prompt.encode("utf-8")) / 3.5   # ~3.5 char/tok
    return approx_tokens * SONNET_INPUT_USD_PER_TOK


def estimate_output_cost_usd(max_tokens: int) -> float:
    """Pre-flight worst-case output cost."""
    return max_tokens * SONNET_OUTPUT_USD_PER_TOK


def build_contract_spec(cap_usd: float):
    """Build AC Contract spec if AC is installed; otherwise local equivalent."""
    if AC_AVAILABLE:
        return Contract(
            id="lang001",
            name="LANG-001 clarification",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(
                tokens=10_000,
                api_calls=50,
                cost_usd=cap_usd,
            ),
        )
    else:
        return {"cap_usd": cap_usd}


def run_one_trial(trial_id: int, contract) -> TrialResult:
    if AC_AVAILABLE:
        cap_usd = contract.resources.cost_usd
    else:
        cap_usd = contract["cap_usd"]
    cap_uc = int(round(cap_usd * 1_000_000))
    start = time.time()

    n_calls = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0
    refused_pre = 0
    refused_reason = ""
    response_text = ""

    # Pre-flight check (this is the equivalent of TB's budget.spend())
    est_input  = estimate_input_cost_usd(LANG_001_PROMPT)
    est_output = estimate_output_cost_usd(MAX_OUTPUT_TOKENS)
    est_call_cost = est_input + est_output

    if est_call_cost > cap_usd:
        refused_pre = 1
        refused_reason = (f"pre-flight: est_call_cost=${est_call_cost:.6f} > "
                          f"cap=${cap_usd:.6f}")
    else:
        # Cap not exceeded by pre-flight estimate; make the call
        try:
            response = litellm.completion(
                model=MODEL_LITELLM,
                messages=[{"role": "user", "content": LANG_001_PROMPT}],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            n_calls += 1
            usage = getattr(response, "usage", None)
            if usage:
                total_in  = getattr(usage, "prompt_tokens", 0) or 0
                total_out = getattr(usage, "completion_tokens", 0) or 0
            total_cost = (total_in * SONNET_INPUT_USD_PER_TOK +
                          total_out * SONNET_OUTPUT_USD_PER_TOK)
            try:
                response_text = response.choices[0].message.content[:200]
            except Exception:
                response_text = "(no content)"
        except Exception as e:
            refused_reason = f"api_error: {type(e).__name__}: {str(e)[:160]}"

    overshoot = 1 if total_cost > cap_usd else 0
    wall = time.time() - start

    return TrialResult(
        trial_id=trial_id, workload="lang001",
        framework="agent_contracts_v0.3.1_manual",
        cap_usd=cap_usd, cap_uc_equiv=cap_uc,
        n_calls=n_calls,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=round(total_cost, 8),
        total_cost_uc_equiv=int(round(total_cost * 1_000_000)),
        overshoot=overshoot,
        refused_pre_flight=refused_pre,
        refused_reason=refused_reason,
        response_truncated=response_text.replace("\n", " ")[:160],
        wall_time_s=round(wall, 3),
    )


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. source ~/.zshrc first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = build_contract_spec(CAP_USD)

    print("=== AC vs TB head-to-head (MANUAL enforcement bypass) ===")
    print(f"    framework: ai-agent-contracts v0.3.1 spec types, manual enforcement")
    print(f"    model:     {MODEL_LITELLM}")
    print(f"    cap:       540 uc (${CAP_USD:.6f})")
    print(f"    workload:  LANG-001 clarification")
    print(f"    n:         {N_TRIALS}")
    print(f"    output:    {OUT_PATH}")

    # Pre-flight estimate for the workload — print once for context
    est_in = estimate_input_cost_usd(LANG_001_PROMPT)
    est_out = estimate_output_cost_usd(MAX_OUTPUT_TOKENS)
    print(f"\n    pre-flight estimate per call:")
    print(f"      input  : ${est_in:.6f} ({int(est_in * 1_000_000)} uc)")
    print(f"      output : ${est_out:.6f} ({int(est_out * 1_000_000)} uc)")
    print(f"      total  : ${est_in+est_out:.6f} ({int((est_in+est_out)*1_000_000)} uc)")
    if est_in + est_out > CAP_USD:
        print(f"      => exceeds cap (${CAP_USD:.6f}); expect pre-flight refusal on all 30 trials")
        print(f"         this would mirror Token Budgets' refusal-to-operate at cap=540 uc")
    print()

    overshoots = 0
    refused_pf = 0
    completed = 0

    with open(OUT_PATH, "w", newline="") as f:
        writer = None
        for trial in range(N_TRIALS):
            print(f"    trial {trial+1}/{N_TRIALS} ", end="", flush=True)
            row = run_one_trial(trial, contract)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
                writer.writeheader()
            writer.writerow(asdict(row))
            f.flush()
            if row.overshoot == 1: overshoots += 1
            if row.refused_pre_flight == 1: refused_pf += 1
            if row.n_calls > 0: completed += 1
            tag = "REFUSED-PF" if row.refused_pre_flight else \
                ("OVER" if row.overshoot else "ok")
            print(f"calls={row.n_calls} cost=${row.total_cost_usd:.6f} [{tag}]")
            time.sleep(0.3)

    print()
    print("SUMMARY")
    print(f"    cost overshoots:       {overshoots}/{N_TRIALS}")
    print(f"    pre-flight refusals:   {refused_pf}/{N_TRIALS}")
    print(f"    completed naturally:   {completed}/{N_TRIALS}")
    print(f"    CSV:                   {OUT_PATH}")


if __name__ == "__main__":
    main()