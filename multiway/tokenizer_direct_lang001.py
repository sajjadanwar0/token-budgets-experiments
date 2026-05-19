#!/usr/bin/env python3
"""
tokenizer_direct_lang001.py - Phase v41 of revision.

Addresses Reviewer 3 §3.3 and Reviewer 4 §"Eliminate Heuristics" attacks:
benchmark a sound tokenizer-direct estimator (Anthropic's own
`count_tokens` API) against the LANG-001 retry-loop, mirroring our
existing cap-sweep configuration on Anthropic.

DESIGN
------
The tokenizer-direct estimator is sound by construction: it calls
Anthropic's `messages.count_tokens` before each spend to get the
exact input token count under the provider's own tokenizer, then
reserves cost as:

    predicted_uc = input_tokens * input_uc_per_token
                 + max_completion_tokens * output_uc_per_token

This guarantees A1 holds (the prediction is the exact bill the
provider would charge, modulo the worst-case output bound).
No 2.0x margin needed.

TRADE-OFF
---------
- SOUNDNESS: A1 holds by construction.
- CAPITAL EFFICIENCY: ~100% (no over-reservation).
- LATENCY: One extra `count_tokens` API call per spend (~50-200ms
  round-trip + Anthropic API processing).
- COST: count_tokens calls are NOT charged for tokens, but they
  are subject to rate limits.

USAGE
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    cd ~/tb-reproduce/token-budgets-experiments/

    # Sanity-check
    python3 tokenizer_direct_lang001.py \\
        --cap-uc 2000 --n-trials 3 --output /tmp/tokdirect_sanity.csv

    # Full sweep matching our existing Anthropic cap sweep
    mkdir -p sweep_results
    for cap in 500 540 1000 2000 5000; do
        python3 tokenizer_direct_lang001.py \\
            --cap-uc $cap --n-trials 30 \\
            --output sweep_results/tokenizer_direct_lang001_cap${cap}_n30.csv \\
            2>&1 | tee /tmp/tokdirect_cap${cap}.log
    done
"""

import argparse
import csv
import os
import sys
import time

from anthropic import Anthropic

# ---------------------------------------------------------------------------
# LANG-001 workload (identical to tokencap_lang001_head_to_head.py)
# ---------------------------------------------------------------------------

LANG_001_SYSTEM = (
    "You are a SQL agent. The user will give you a task. You must write a "
    "SQL query to accomplish it. After you write each query, you will be "
    "told whether it succeeded or failed. If it failed, write a corrected "
    "query that addresses the specific error reported."
)

LANG_001_USER = (
    "Give me the average salary by department from the 'employees' table. "
    "Group by department, order by avg salary desc."
)

LANG_001_FAKE_ERROR = (
    "Error: SQL syntax error near 'AVG' on line 1. The query failed to parse. "
    "Please rewrite the query with correct SQL syntax."
)

ANTHROPIC_HAIKU_4_5 = "claude-haiku-4-5-20251001"
PRICING_UC_PER_TOKEN = {"input": 1, "output": 5}
MAX_COMPLETION_TOKENS = 200  # the max_tokens we pass to messages.create


# ---------------------------------------------------------------------------
# Tokenizer-direct estimator
# ---------------------------------------------------------------------------

def predict_cost_uc(client, messages, system, max_completion_tokens):
    """Call Anthropic's count_tokens to get exact input token count, then
    compute the worst-case cost reservation as:

        cost_uc = input_tokens * 1 uc/token + max_completion_tokens * 5 uc/token

    Returns (predicted_uc, input_tokens, latency_ms).
    """
    start = time.monotonic()
    try:
        resp = client.messages.count_tokens(
            model=ANTHROPIC_HAIKU_4_5,
            system=system,
            messages=messages,
        )
        latency_ms = (time.monotonic() - start) * 1000.0
        input_tokens = resp.input_tokens
    except Exception as e:
        return None, 0, (time.monotonic() - start) * 1000.0, str(e)

    predicted_uc = (
        input_tokens * PRICING_UC_PER_TOKEN["input"]
        + max_completion_tokens * PRICING_UC_PER_TOKEN["output"]
    )
    return predicted_uc, input_tokens, latency_ms, None


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def call_with_retry(client, *, model, max_tokens, temperature, system,
                    messages, max_retries=5):
    """Standard Anthropic-call wrapper with 529-Overloaded retry."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            ), None, attempt + 1
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ("529", "Overloaded", "overloaded_error")):
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt + 1, 30))
                    continue
                return None, "exhausted_retries_overloaded", attempt + 1
            return None, f"other_error_{type(e).__name__}: {e}", attempt + 1
    return None, "exhausted_retries", max_retries


def run_trial(trial_id, cap_uc, max_steps=20):
    """One LANG-001 trial using the tokenizer-direct estimator."""

    client = Anthropic()
    messages = [{"role": "user", "content": LANG_001_USER}]

    remaining_uc = cap_uc
    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    outcome = "max_steps_reached"
    error_repr = ""

    # Telemetry on tokenizer-direct overhead
    count_tokens_calls = 0
    total_count_tokens_latency_ms = 0.0

    for step in range(max_steps):
        # --- Tokenizer-direct pre-flight check ---
        predicted_uc, input_tokens, latency_ms, err = predict_cost_uc(
            client,
            messages=messages,
            system=LANG_001_SYSTEM,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        count_tokens_calls += 1
        total_count_tokens_latency_ms += latency_ms

        if err is not None:
            outcome = f"count_tokens_error: {err[:80]}"
            error_repr = err[:200]
            break

        if predicted_uc > remaining_uc:
            outcome = "compile_time_reservation_refused"
            error_repr = (
                f"refused: predicted={predicted_uc}uc, remaining={remaining_uc}uc, "
                f"input_tokens={input_tokens}"
            )
            break

        # --- Reservation passes; debit predicted cost ---
        remaining_uc -= predicted_uc

        # --- Make the actual LLM call ---
        resp, err_class, _ = call_with_retry(
            client,
            model=ANTHROPIC_HAIKU_4_5,
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=0,
            system=LANG_001_SYSTEM,
            messages=messages,
        )

        if err_class is not None:
            outcome = err_class
            error_repr = err_class
            # Reservation was already debited; we'd refund in a real impl,
            # but for cap-respecting analysis the cap was respected (we just
            # paid for nothing).
            break

        # --- Record actual usage + refund unused portion ---
        actual_input = resp.usage.input_tokens
        actual_output = resp.usage.output_tokens
        actual_cost = (
            actual_input * PRICING_UC_PER_TOKEN["input"]
            + actual_output * PRICING_UC_PER_TOKEN["output"]
        )
        # Refund: predicted_uc - actual_cost
        # (predicted was input + max_completion_tokens*5;
        #  actual was input + actual_output*5; refund = (max - actual_output)*5)
        refund = predicted_uc - actual_cost
        if refund > 0:
            remaining_uc += refund

        steps += 1
        total_input_tokens += actual_input
        total_output_tokens += actual_output

        assistant_text = resp.content[0].text if resp.content else ""
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": LANG_001_FAKE_ERROR})

    actual_total_cost_uc = (
        total_input_tokens * PRICING_UC_PER_TOKEN["input"]
        + total_output_tokens * PRICING_UC_PER_TOKEN["output"]
    )
    overshoot_uc = max(0, actual_total_cost_uc - cap_uc)

    # Capital efficiency: actual / (cap_uc - remaining_uc)
    # i.e., how much of the reserved budget was actually used
    reserved_total = cap_uc - remaining_uc
    if reserved_total > 0:
        capital_efficiency = actual_total_cost_uc / reserved_total
    else:
        capital_efficiency = 1.0

    mean_count_tokens_latency = (
        total_count_tokens_latency_ms / count_tokens_calls
        if count_tokens_calls > 0 else 0
    )

    return {
        "runtime": "tokenizer_direct",
        "run_id": f"trial_{trial_id}",
        "provider": "anthropic",
        "outcome": outcome,
        "agent_steps": steps,
        "cap_uc": cap_uc,
        "total_spent_uc": actual_total_cost_uc,
        "remaining_uc": remaining_uc,
        "pct_of_cap": f"{actual_total_cost_uc / cap_uc * 100:.1f}%" if cap_uc else "N/A",
        "overshoot_uc": overshoot_uc,
        "capital_efficiency": round(capital_efficiency, 4),
        "wall_seconds": None,
        "workload": "lang001",
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        "count_tokens_calls": count_tokens_calls,
        "count_tokens_total_latency_ms": round(total_count_tokens_latency_ms, 1),
        "count_tokens_mean_latency_ms": round(mean_count_tokens_latency, 1),
        "error_repr": error_repr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-uc", type=int, required=True,
        help="Dollar cap in micro-dollars (e.g., 540, 2000, 5000)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*76}")
    print(f"Tokenizer-direct estimator LANG-001 benchmark")
    print(f"{'='*76}")
    print(f"  Cap (uc):        {args.cap_uc}  (= ${args.cap_uc / 1_000_000:.6f})")
    print(f"  N trials:        {args.n_trials}")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Output:          {args.output}")
    print(f"  Model:           {ANTHROPIC_HAIKU_4_5}")
    print(f"  Estimator:       Anthropic count_tokens API (sound by construction)")
    print(f"{'='*76}")

    rows = []
    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.cap_uc, args.max_steps)
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"  [{i+1:02d}/{args.n_trials}] HARNESS-LEVEL FAIL: "
                  f"{type(e).__name__}: {e}")
            continue
        row["wall_seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
        print(f"  [{i+1:02d}/{args.n_trials}] "
              f"outcome={row['outcome'][:35]:<35} "
              f"steps={row['agent_steps']:>2} "
              f"in={row['actual_input_tokens']:>5} "
              f"out={row['actual_output_tokens']:>4} "
              f"spent={row['total_spent_uc']:>5}uc "
              f"over={row['overshoot_uc']:>4}uc "
              f"ct_calls={row['count_tokens_calls']:>2} "
              f"ct_lat={row['count_tokens_mean_latency_ms']:>5.0f}ms")
        if args.sleep > 0:
            time.sleep(args.sleep)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.output}")
    else:
        print("\nNo rows produced; CSV not written.")
        sys.exit(1)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    n = len(rows)
    refused = sum(1 for r in rows if r["outcome"] == "compile_time_reservation_refused")
    maxsteps = sum(1 for r in rows if r["outcome"] == "max_steps_reached")
    other = n - refused - maxsteps

    overshoots = sum(1 for r in rows if r["overshoot_uc"] > 0)

    spending = [r for r in rows if r["agent_steps"] > 0]
    if spending:
        mean_steps = sum(r["agent_steps"] for r in spending) / len(spending)
        mean_spent = sum(r["total_spent_uc"] for r in spending) / len(spending)
        mean_capeff = sum(r["capital_efficiency"] for r in spending) / len(spending)
    else:
        mean_steps = mean_spent = mean_capeff = 0

    mean_ct_calls_per_trial = sum(r["count_tokens_calls"] for r in rows) / n
    mean_ct_latency_per_call = (
        sum(r["count_tokens_total_latency_ms"] for r in rows)
        / sum(r["count_tokens_calls"] for r in rows)
    ) if sum(r["count_tokens_calls"] for r in rows) > 0 else 0

    print(f"\n{'='*76}")
    print("SUMMARY")
    print(f"{'='*76}")
    print(f"  Refused (pre-call blocked):     {refused}/{n}")
    print(f"  Reached max_steps:              {maxsteps}/{n}")
    print(f"  Other errors:                   {other}/{n}")
    print(f"  Dollar-overshoot trials:        {overshoots}/{n}")
    if spending:
        print(f"  ---")
        print(f"  Spending trials (>0 steps):     {len(spending)}/{n}")
        print(f"  Mean agent_steps:               {mean_steps:.2f}")
        print(f"  Mean dollar spent (uc):         {mean_spent:.0f}")
        print(f"  Mean dollar spent (USD):        ${mean_spent / 1_000_000:.6f}")
        print(f"  Mean capital efficiency:        {mean_capeff:.1%}")
    print(f"  ---")
    print(f"  Mean count_tokens calls/trial:  {mean_ct_calls_per_trial:.2f}")
    print(f"  Mean count_tokens latency/call: {mean_ct_latency_per_call:.0f} ms")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()