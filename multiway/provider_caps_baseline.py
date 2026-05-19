#!/usr/bin/env python3
"""
provider_caps_baseline.py - Provider-side per-call cap baseline.

External review challenge: "the obvious alternative is the provider's own
max_completion_tokens parameter. Show whether TB adds anything over what's
already free." This script answers that challenge directly.

The provider's max_completion_tokens (Anthropic) and max_tokens (OpenAI)
parameters bound OUTPUT TOKENS PER CALL. They do NOT bound CUMULATIVE
SESSION SPEND. This script measures whether per-call output-bounding
alone is sufficient to keep cumulative session spend below a target cap,
or whether session-level mechanisms (TB pre-flight, gateway post-hoc,
tokencap, etc) are needed.

Protocol:
    Run LANG-001 retry-loop with max_completion_tokens=200, no
    session-level cap. Measure cumulative spend after max_steps. Compare
    to the same 5 caps used in Tables 37, 38, 42 (gateway baseline).

Expected finding: max_completion_tokens alone overshoots every reasonable
session cap because the agent loops to max_steps; total spend = sum of
per-call costs, with no mechanism halting the loop based on cumulative
dollars. This confirms the paper's framing (which is currently buried).

USAGE
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    cd ~/tb-reproduce/token-budgets-experiments/

    # Sanity (N=3)
    python3 provider_caps_baseline.py --n-trials 3 \\
        --output /tmp/provider_caps_sanity.csv

    # Full sweep: N=30 trials, ~$0.50, ~10 min
    mkdir -p sweep_results
    python3 provider_caps_baseline.py --n-trials 30 \\
        --output sweep_results/provider_caps_baseline_lang001_n30.csv
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
MAX_COMPLETION_TOKENS = 200
TARGET_CAPS = [500, 540, 1000, 2000, 5000]  # for comparison vs Tables 37/38/42


def call_with_retry(client, *, model, max_tokens, temperature, system,
                    messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model=model, max_tokens=max_tokens,
                temperature=temperature, system=system, messages=messages,
            ), None
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ("529", "Overloaded", "overloaded_error")):
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt + 1, 30))
                    continue
                return None, "exhausted_retries_overloaded"
            return None, f"other_error_{type(e).__name__}: {e}"
    return None, "exhausted_retries"


def run_trial(trial_id, max_steps=20):
    """LANG-001 with max_completion_tokens=200 and NO session-level cap."""
    client = Anthropic()
    messages = [{"role": "user", "content": LANG_001_USER}]

    cumulative_spent_uc = 0
    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    per_step_spend = []
    outcome = "max_steps_reached"
    error_repr = ""

    for step in range(max_steps):
        resp, err_class = call_with_retry(
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
        actual_cost = (actual_input * PRICING_UC_PER_TOKEN["input"]
                       + actual_output * PRICING_UC_PER_TOKEN["output"])
        cumulative_spent_uc += actual_cost
        per_step_spend.append(actual_cost)

        steps += 1
        total_input_tokens += actual_input
        total_output_tokens += actual_output

        assistant_text = resp.content[0].text if resp.content else ""
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": LANG_001_FAKE_ERROR})

    # Compute overshoot vs each target cap
    overshoots_per_cap = {cap: cumulative_spent_uc > cap for cap in TARGET_CAPS}

    return {
        "runtime": "provider_caps_only",
        "run_id": f"trial_{trial_id}",
        "provider": "anthropic",
        "outcome": outcome,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_steps": max_steps,
        "agent_steps_executed": steps,
        "total_spent_uc": cumulative_spent_uc,
        "per_step_spend_uc": ",".join(str(s) for s in per_step_spend),
        "wall_seconds": None,
        "workload": "lang001",
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        **{f"over_cap_{cap}": int(overshoots_per_cap[cap]) for cap in TARGET_CAPS},
        "error_repr": error_repr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
    print(f"Provider-side per-call cap baseline (max_completion_tokens only)")
    print(f"{'='*76}")
    print(f"  Model:                 {ANTHROPIC_HAIKU_4_5}")
    print(f"  max_completion_tokens: {MAX_COMPLETION_TOKENS}")
    print(f"  max_steps:             {args.max_steps}")
    print(f"  N trials:              {args.n_trials}")
    print(f"  Session cap:           NONE (this is the test)")
    print(f"  Target caps for comparison: {TARGET_CAPS}")
    print(f"{'='*76}")

    rows = []
    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.max_steps)
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"  [{i+1:02d}/{args.n_trials}] HARNESS FAIL: {type(e).__name__}: {e}")
            continue
        row["wall_seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
        over_flags = "".join("Y" if row[f"over_cap_{c}"] else "n" for c in TARGET_CAPS)
        print(f"  [{i+1:02d}/{args.n_trials}] "
              f"steps={row['agent_steps_executed']:>2} "
              f"spent={row['total_spent_uc']:>5}uc "
              f"over_caps[{'/'.join(str(c) for c in TARGET_CAPS)}]={over_flags}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.output}")

    n = len(rows)
    if n == 0:
        return
    mean_steps = sum(r["agent_steps_executed"] for r in rows) / n
    mean_spend = sum(r["total_spent_uc"] for r in rows) / n
    min_spend = min(r["total_spent_uc"] for r in rows)
    max_spend = max(r["total_spent_uc"] for r in rows)

    print(f"\n{'='*76}")
    print(f"SUMMARY: provider per-call cap only, N={n}")
    print(f"{'='*76}")
    print(f"  Mean steps executed:    {mean_steps:.2f} (out of max={args.max_steps})")
    print(f"  Spend (uc):             mean={mean_spend:.0f}, min={min_spend}, max={max_spend}")
    print(f"  Overshoot rate vs each target cap:")
    for cap in TARGET_CAPS:
        overshoot_count = sum(r[f"over_cap_{cap}"] for r in rows)
        mean_overshoot = sum(max(0, r["total_spent_uc"] - cap) for r in rows) / n
        ratio = mean_spend / cap
        print(f"    cap={cap:>5}uc: {overshoot_count:>2}/{n} overshoot, "
              f"mean ratio = {ratio:.2f}x (mean overshoot = {mean_overshoot:.0f} uc)")
    print(f"{'='*76}")
    print(f"")
    print(f"Interpretation: max_completion_tokens bounds OUTPUT PER CALL only.")
    print(f"It does NOT bound CUMULATIVE SESSION SPEND. If max spend > any target")
    print(f"cap, then max_completion_tokens alone is insufficient for that cap.")


if __name__ == "__main__":
    main()