"""
m2_runtime_adapters.py

Two runtime adapters for the M2 head-to-head experiment:

  1. run_tb_python_byte_length: the existing TB Python port wired to use the
     SAME byte-length + 2.0x margin estimator as the Rust impl. This isolates
     the type-system contribution from the estimator-choice contribution that
     confounds the Table 8 comparison.

  2. run_naive_pre_flight_guard: a 4-line runtime guard with no library, no
     affine type, no _consumed flag. Module-level remaining-uc counter,
     byte-length estimate before each call, raises BudgetExceeded if the
     estimate would push spending over cap.

Both adapters return per-trial outcome dicts in the same shape multiway_compare.py
already consumes for the existing runtimes:

    [
      {"trial": int, "spent_uc": int, "overshoot_uc": int, "outcome": str},
      ...
    ]

outcome codes follow the existing convention:
  "T"  = cap respected, pre-flight reservation refused a call
  "C"  = completed naturally within cap (no refusal needed)
  "OS" = overshoot (cap exceeded)
  "E"  = error (provider error, parsing failure, etc.)
"""

from __future__ import annotations

from typing import Callable, List, Dict, Any

# ---------------------------------------------------------------------------
# Shared estimator: byte-length input + max_output_tokens output, with the
# 2.0x Anthropic safety margin. This MUST match the Rust implementation's
# AnthropicEstimator and ByteLengthEstimator so the M2 comparison is on
# estimator-equal footing.
# ---------------------------------------------------------------------------

# Per-token prices in micro-cents (uc), 1 uc = 10^-5 USD.
# Mirrors token-budgets/src/estimator/default.rs.
PROVIDER_PRICES_UC_PER_TOKEN = {
    "openai-gpt-4o":               {"in":  250, "out": 1000},   # $2.50 / $10.00 per Mtok
    "openai-gpt-4o-mini":          {"in":   15, "out":   60},   # $0.15 / $0.60 per Mtok
    "anthropic-claude-sonnet-4-5": {"in":  300, "out": 1500},   # $3.00 / $15.00 per Mtok
    "anthropic-claude-haiku-4-5":  {"in":  100, "out":  500},   # $1.00 / $5.00 per Mtok
    "groq-llama-3.3-70b":          {"in":   59, "out":   79},
}

ANTHROPIC_SAFETY_MARGIN = 2.0


def estimate_call_uc(prompt: str, max_output_tokens: int, provider_key: str) -> int:
    """Conservative pre-flight estimate in uc, matching the Rust impl.

    Input side: byte-length of prompt, multiplied by safety margin on Anthropic.
    Output side: full max_output_tokens reservation at the per-output-token rate.
    """
    prices = PROVIDER_PRICES_UC_PER_TOKEN[provider_key]
    input_bytes = len(prompt.encode("utf-8"))
    margin = ANTHROPIC_SAFETY_MARGIN if provider_key.startswith("anthropic-") else 1.0
    input_uc = int(input_bytes * prices["in"] * margin / 1_000_000.0 * 1_000_000.0)
    # The /1e6 then *1e6 above looks redundant but preserves integer semantics
    # matching the Rust checked_mul; left explicit for cross-checking with
    # token-budgets/src/estimator/default.rs.
    input_uc_clean = (input_bytes * prices["in"] * (2 if margin == 2.0 else 1))
    # Use micro-cents directly: prices are uc-per-Mtok, bytes are <=tokens.
    # Conservative: 1 byte <= 1 token under all tested tokenizers (with margin).
    input_uc = int(input_bytes * prices["in"] / 1000) * (2 if margin == 2.0 else 1)
    output_uc = int(max_output_tokens * prices["out"] / 1000)
    return input_uc + output_uc


def actual_call_uc(input_tokens: int, output_tokens: int, provider_key: str) -> int:
    """Post-call actual cost in uc."""
    prices = PROVIDER_PRICES_UC_PER_TOKEN[provider_key]
    return int(input_tokens * prices["in"] / 1000) + int(output_tokens * prices["out"] / 1000)


# ---------------------------------------------------------------------------
# Adapter 1: TB Python port with byte-length estimator (NO type system)
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when a pre-flight reservation would exceed the cap."""
    pass


def run_tb_python_byte_length(
        provider: str,
        model: str,
        workload: str,
        cap_uc: int,
        n_replicas: int,
        seed: int = 42,
        llm_caller: Callable[..., Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """TB Python port adapter using byte-length+2x margin estimation.

    The Python port has NO compile-time integrity: it uses a runtime
    _consumed flag to enforce affine semantics at the Python level. The
    estimator is exactly the same as the Rust impl, so the only variable
    isolated by this row vs. tb_rust_impl is type-system enforcement.
    """
    from token_budgets import Budget

    provider_key = f"{provider}-{model}".replace("claude-", "")
    if provider_key not in PROVIDER_PRICES_UC_PER_TOKEN:
        # Try common normalisations.
        for k in PROVIDER_PRICES_UC_PER_TOKEN:
            if model in k or k in model:
                provider_key = k
                break
        else:
            raise ValueError(f"unknown provider/model: {provider}/{model}")

    results: List[Dict[str, Any]] = []

    for trial in range(n_replicas):
        budget = Budget(initial_uc=cap_uc, max_uc=cap_uc)
        spent = 0
        outcome = "C"
        try:
            messages = workload_initial_prompt(workload)
            for step in range(20):   # max_steps cap
                prompt = render_prompt(messages, workload)
                est = estimate_call_uc(prompt, 200, provider_key)
                if est > budget.available():
                    outcome = "T"
                    break
                # Decrement reservation through the Python port's _consumed
                # flag discipline (runtime affine enforcement, NO type system).
                receipt = budget.spend(est)
                if llm_caller is None:
                    response = _live_llm_call(provider, model, prompt, max_tokens=200)
                else:
                    response = llm_caller(provider, model, prompt, max_tokens=200)
                actual = actual_call_uc(
                    response["usage"]["input_tokens"],
                    response["usage"]["output_tokens"],
                    provider_key,
                )
                receipt.confirm(actual_charge=actual)
                spent += actual
                messages = update_messages(messages, response, workload)
                if workload_terminated(messages, workload):
                    outcome = "C"
                    break
        except BudgetExceeded:
            outcome = "T"
        except Exception as e:
            outcome = "E"
            results.append({
                "trial": trial, "spent_uc": spent, "overshoot_uc": 0,
                "outcome": outcome, "error": str(e),
            })
            continue

        overshoot = max(0, spent - cap_uc)
        if overshoot > 0 and outcome != "OS":
            outcome = "OS"
        results.append({
            "trial": trial,
            "spent_uc": spent,
            "overshoot_uc": overshoot,
            "outcome": outcome,
        })

    return results


# ---------------------------------------------------------------------------
# Adapter 2: Naive 4-line pre-flight guard (NO library, NO type system)
# ---------------------------------------------------------------------------

def run_naive_pre_flight_guard(
        provider: str,
        model: str,
        workload: str,
        cap_uc: int,
        n_replicas: int,
        seed: int = 42,
        llm_caller: Callable[..., Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Bare 4-line pre-flight guard, no library.

    The discipline:
        remaining = cap_uc
        for each call:
            est = estimate_call_uc(...)
            if est > remaining: raise BudgetExceeded
            remaining -= est           # decrement on reservation
            response = llm_call(...)
            remaining += (est - actual)  # refund difference

    No affine type, no _consumed flag, no library. Just a local counter and
    a guard. If THIS achieves 0/30, the cap-respecting outcome is supplied
    entirely by pre-flight timing.
    """
    provider_key = f"{provider}-{model}".replace("claude-", "")
    if provider_key not in PROVIDER_PRICES_UC_PER_TOKEN:
        for k in PROVIDER_PRICES_UC_PER_TOKEN:
            if model in k or k in model:
                provider_key = k
                break
        else:
            raise ValueError(f"unknown provider/model: {provider}/{model}")

    results: List[Dict[str, Any]] = []

    for trial in range(n_replicas):
        remaining = cap_uc
        spent = 0
        overshoot = 0
        outcome = "C"
        try:
            messages = workload_initial_prompt(workload)
            for step in range(20):
                prompt = render_prompt(messages, workload)
                est = estimate_call_uc(prompt, 200, provider_key)

                # ============ THE 4-LINE GUARD ============
                if est > remaining:
                    outcome = "T"
                    break
                remaining -= est
                # ==========================================

                if llm_caller is None:
                    response = _live_llm_call(provider, model, prompt, max_tokens=200)
                else:
                    response = llm_caller(provider, model, prompt, max_tokens=200)
                actual = actual_call_uc(
                    response["usage"]["input_tokens"],
                    response["usage"]["output_tokens"],
                    provider_key,
                )
                refund = est - actual
                if refund > 0:
                    remaining += refund

                spent += actual
                messages = update_messages(messages, response, workload)
                if workload_terminated(messages, workload):
                    outcome = "C"
                    break
        except Exception as e:
            outcome = "E"
            results.append({
                "trial": trial, "spent_uc": spent, "overshoot_uc": 0,
                "outcome": outcome, "error": str(e),
            })
            continue

        overshoot = max(0, spent - cap_uc)
        if overshoot > 0:
            outcome = "OS"
        results.append({
            "trial": trial,
            "spent_uc": spent,
            "overshoot_uc": overshoot,
            "outcome": outcome,
        })

    return results


# ---------------------------------------------------------------------------
# Workload helpers: shared with the existing multiway_compare.py.
# If these names collide with existing definitions in multiway_compare.py,
# delete them here and import them from there instead.
# ---------------------------------------------------------------------------

def workload_initial_prompt(workload: str) -> list:
    """Returns the initial message list for the named workload (LANG-001 etc).

    The existing multiway_compare.py has these definitions. If you are
    pasting this file as a standalone module, copy the LANG-001 setup from
    multiway_compare.py into this function body.
    """
    if workload == "lang001":
        return [
            {"role": "system", "content":
                "You are a SQL assistant. Use the run_sql tool to query the database."},
            {"role": "user", "content":
                "Find all employees in the engineering department earning more than $100k."},
        ]
    raise NotImplementedError(f"workload {workload!r} not defined; copy from multiway_compare.py")


def render_prompt(messages: list, workload: str) -> str:
    """Serialise the message list back into a single prompt string for byte-length
    estimation. Matches what the LLM client actually sends on the wire."""
    return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)


def update_messages(messages: list, response: Dict[str, Any], workload: str) -> list:
    """Append the assistant response and any tool-call results to the message list."""
    new = list(messages)
    new.append({"role": "assistant", "content": response.get("content", "")})
    if "tool_calls" in response:
        for tc in response["tool_calls"]:
            new.append({"role": "tool", "content": f"(simulated result for {tc['name']})"})
    return new


def workload_terminated(messages: list, workload: str) -> bool:
    """Returns True if the workload self-terminated (model did not request another tool)."""
    last = messages[-1] if messages else {}
    return last.get("role") == "assistant" and "tool_calls" not in last


# ---------------------------------------------------------------------------
# Live LLM call. Use whatever client multiway_compare.py already imports
# (anthropic, openai). If this collides with an existing helper, delete this
# and import _live_llm_call from multiway_compare.py instead.
# ---------------------------------------------------------------------------

def _live_llm_call(provider: str, model: str, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return {
            "content": resp.choices[0].message.content or "",
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "content": resp.content[0].text if resp.content else "",
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        }
    else:
        raise ValueError(f"unknown provider: {provider}")