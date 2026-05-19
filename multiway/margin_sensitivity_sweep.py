#!/usr/bin/env python3
"""
margin_sensitivity_sweep.py - Priority A item 5 of v43 revision.

Addresses the brutal-review attack: "Why 2.0× margin? No sensitivity
analysis is given." This script runs the byte-length estimator with
varying safety margins {1.0, 1.5, 2.0, 2.5, 3.0} on the LANG-001
workload, measuring overshoot rate and capital efficiency at each
margin.

DESIGN
------
This is a Python re-implementation of the byte-length + margin
estimator used in the Rust crate's AnthropicEstimator. It mirrors:

    predicted_uc = byte_length(rendered_prompt) * margin * input_pricing
                 + max_completion_tokens * output_pricing

The same LANG-001 retry-loop workload is used as the existing
sweeps so results are directly comparable to Table 38
(tokenizer-direct) and the Table 33 cap-sweep.

USAGE
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    # Sanity (3 trials)
    python3 margin_sensitivity_sweep.py --margin 2.0 --cap-uc 2000 \\
        --n-trials 3 --output /tmp/margin_sanity.csv

    # Full sweep: 5 margins x 1 cap (=2000) x N=15 = 75 trials, ~$1
    mkdir -p sweep_results
    for margin in 1.0 1.5 2.0 2.5 3.0; do
        python3 margin_sensitivity_sweep.py \\
            --margin $margin --cap-uc 2000 --n-trials 15 \\
            --output sweep_results/margin_sensitivity_margin${margin}_cap2000_n15.csv
        sleep 5
    done
"""

import argparse
import csv
import os
import sys
import time

from anthropic import Anthropic

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
# 100 matches the Rust harness setup (Table 32 sweep used max_tokens≈100;
# see runner.py and tb_rust_anthropic_cap2000_n30.csv where output_uc=511
# implies max_completion_tokens ≈ 102).
MAX_COMPLETION_TOKENS = 100


def serialize_request_body(messages, system, max_completion_tokens):
    """Mirror the deployed Rust AnthropicEstimator: serialize the request
    body to UTF-8 JSON exactly like the Anthropic client would, since
    that is what the Rust crate's ByteLength estimator measures.
    From runner.py: 'sort_keys=True, separators=(",", ":")'."""
    import json
    payload = {
        "model": ANTHROPIC_HAIKU_4_5,
        "max_tokens": max_completion_tokens,
        "system": system,
        "messages": messages,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def predict_cost_bytelen(messages, system, max_completion_tokens, margin):
    """Byte-length estimator with safety margin, matching the deployed
    Rust AnthropicEstimator<ByteLength>::estimate exactly:

        input_uc  = ceil(byte_length(request_body) × safety_margin)
        output_uc = max_completion_tokens × output_pricing  (NO margin)
        total_uc  = input_uc + output_uc

    The byte_length is treated 1:1 as input tokens (pessimistic by ~4x
    for English; this *is* the source of the discipline's conservatism,
    NOT an additional /4 conversion).

    Source: token-budgets/src/estimator.rs lines 217-223.
    """
    import math
    request_body = serialize_request_body(messages, system, max_completion_tokens)
    byte_len = len(request_body.encode("utf-8"))
    # Bytes treated 1:1 as input tokens at $1/M input pricing
    predicted_input_uc = math.ceil(byte_len * margin * PRICING_UC_PER_TOKEN["input"])
    # Output is already hard-capped by max_completion_tokens; no margin needed
    predicted_output_uc = max_completion_tokens * PRICING_UC_PER_TOKEN["output"]
    return predicted_input_uc + predicted_output_uc


def call_with_retry(client, *, model, max_tokens, temperature, system,
                    messages, max_retries=5):
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


def run_trial(trial_id, cap_uc, margin, max_steps=20):
    client = Anthropic()
    messages = [{"role": "user", "content": LANG_001_USER}]

    remaining_uc = cap_uc
    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    outcome = "max_steps_reached"
    error_repr = ""

    for step in range(max_steps):
        # Byte-length pre-flight prediction
        predicted_uc = predict_cost_bytelen(
            messages, LANG_001_SYSTEM, MAX_COMPLETION_TOKENS, margin
        )

        if predicted_uc > remaining_uc:
            outcome = "compile_time_reservation_refused"
            error_repr = (
                f"refused: predicted={predicted_uc}uc, remaining={remaining_uc}uc"
            )
            break

        remaining_uc -= predicted_uc

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
            break

        actual_input = resp.usage.input_tokens
        actual_output = resp.usage.output_tokens
        actual_cost = (
            actual_input * PRICING_UC_PER_TOKEN["input"]
            + actual_output * PRICING_UC_PER_TOKEN["output"]
        )
        # Refund unused reservation
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

    reserved_total = cap_uc - remaining_uc
    if reserved_total > 0:
        capital_efficiency = actual_total_cost_uc / reserved_total
    else:
        capital_efficiency = 1.0

    return {
        "runtime": "tb_bytelen_python",
        "margin": margin,
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
        "error_repr": error_repr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin", type=float, required=True,
        help="Safety margin multiplier (1.0 = no margin, 2.0 = current default)")
    parser.add_argument("--cap-uc", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=15)
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
    print(f"Margin sensitivity sweep on LANG-001")
    print(f"{'='*76}")
    print(f"  Margin:      {args.margin}x")
    print(f"  Cap (uc):    {args.cap_uc}  (= ${args.cap_uc / 1_000_000:.6f})")
    print(f"  N trials:    {args.n_trials}")
    print(f"  Output:      {args.output}")
    print(f"  Model:       {ANTHROPIC_HAIKU_4_5}")
    print(f"{'='*76}")

    rows = []
    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.cap_uc, args.margin, args.max_steps)
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
              f"spent={row['total_spent_uc']:>5}uc "
              f"over={row['overshoot_uc']:>4}uc "
              f"capeff={row['capital_efficiency']:.2f}")
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
        print("\nNo rows produced.")
        sys.exit(1)

    n = len(rows)
    overshoots = sum(1 for r in rows if r["overshoot_uc"] > 0)
    refused = sum(1 for r in rows if r["outcome"] == "compile_time_reservation_refused")
    spending = [r for r in rows if r["agent_steps"] > 0]

    print(f"\n{'='*76}")
    print(f"SUMMARY: margin={args.margin}x, cap={args.cap_uc}uc, N={n}")
    print(f"{'='*76}")
    print(f"  Refused:               {refused}/{n}")
    print(f"  Dollar-overshoot:      {overshoots}/{n}")
    if spending:
        mean_steps = sum(r["agent_steps"] for r in spending) / len(spending)
        mean_spent = sum(r["total_spent_uc"] for r in spending) / len(spending)
        mean_capeff = sum(r["capital_efficiency"] for r in spending) / len(spending)
        print(f"  Spending trials:       {len(spending)}/{n}")
        print(f"  Mean steps (spending): {mean_steps:.2f}")
        print(f"  Mean spent (uc):       {mean_spent:.0f}")
        print(f"  Mean capital eff:      {mean_capeff:.1%}")
    else:
        print(f"  No trials admitted at least one call (margin too aggressive)")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()