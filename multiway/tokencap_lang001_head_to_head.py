#!/usr/bin/env python3
"""
tokencap_lang001_head_to_head.py - Phase 2 of Path Y revision (v3).

v3 fixes the trial-isolation bug discovered in v2:
  - tokencap.wrap() shares a process-global Guard across calls.
    Without teardown between trials, all trials after #1 inherit the
    Guard state from trial #1.
  - Call tokencap.teardown() at the start of each trial to start
    a fresh session.

USAGE
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tokencap_lang001_head_to_head.py \\
        --token-limit 540 --n-trials 30 \\
        --output sweep_results/tokencap_lang001_limit540_n30.csv
"""

import argparse
import csv
import os
import sys
import time

import tokencap
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# LANG-001 workload
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


def reset_tokencap():
    """Tear down any active Guard so wrap() creates a fresh session.

    Tolerates "no Guard active" on the first call.
    """
    try:
        tokencap.teardown()
    except Exception:
        pass


def make_wrapped_client(token_limit):
    reset_tokencap()
    return tokencap.wrap(Anthropic(), limit=token_limit)


def call_with_retry(client, *, model, max_tokens, temperature, system,
                    messages, max_retries=5):
    """Call client.messages.create with retry on Anthropic 529 Overloaded.

    Returns (response, error_classification, attempts).
    """
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return resp, None, attempt + 1
        except TypeError as e:
            # tokencap 0.2.0 known bug: blocks via TypeError instead of
            # BudgetExceededError. Safety holds (no API call made).
            if "isinstance" in str(e):
                return None, "budget_block_typeerror", attempt + 1
            raise
        except Exception as e:
            ename = type(e).__name__
            if ename == "BudgetExceededError":
                return None, "budget_block_exception", attempt + 1

            err_str = str(e)
            if any(x in err_str for x in ("529", "Overloaded", "overloaded_error")):
                if attempt < max_retries - 1:
                    sleep_s = min(2 ** attempt + 1, 30)
                    time.sleep(sleep_s)
                    continue
                return None, "exhausted_retries_overloaded", attempt + 1

            return None, f"other_error_{ename}", attempt + 1

    return None, "exhausted_retries", max_retries


def run_trial(trial_id, token_limit, max_steps=20):
    """Run one isolated LANG-001 trial with a fresh tokencap session."""

    client = make_wrapped_client(token_limit)
    messages = [{"role": "user", "content": LANG_001_USER}]

    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    outcome = "max_steps_reached"
    error_repr = ""
    retry_count_sum = 0

    for step in range(max_steps):
        resp, err_class, attempts = call_with_retry(
            client,
            model=ANTHROPIC_HAIKU_4_5,
            max_tokens=200,
            temperature=0,
            system=LANG_001_SYSTEM,
            messages=messages,
        )
        retry_count_sum += (attempts - 1)

        if err_class in ("budget_block_typeerror", "budget_block_exception"):
            outcome = "compile_time_reservation_refused"
            error_repr = err_class
            break

        if err_class is not None:
            outcome = err_class
            error_repr = err_class
            break

        steps += 1
        total_input_tokens += resp.usage.input_tokens
        total_output_tokens += resp.usage.output_tokens

        assistant_text = resp.content[0].text if resp.content else ""
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": LANG_001_FAKE_ERROR})

    actual_cost_uc = (
        total_input_tokens * PRICING_UC_PER_TOKEN["input"]
        + total_output_tokens * PRICING_UC_PER_TOKEN["output"]
    )

    cap_uc_lower_bound = token_limit * PRICING_UC_PER_TOKEN["input"]
    overshoot_uc = max(0, actual_cost_uc - cap_uc_lower_bound)

    total_tokens = total_input_tokens + total_output_tokens
    token_overshoot = max(0, total_tokens - token_limit)

    # tokencap's view of post-trial state (informational only)
    tokencap_view = ""
    try:
        status = tokencap.get_status()
        tokencap_view = repr(status)[:300]
    except Exception:
        tokencap_view = ""

    return {
        "runtime": "tokencap",
        "run_id": f"trial_{trial_id}",
        "provider": "anthropic",
        "outcome": outcome,
        "agent_steps": steps,
        "cap_uc": cap_uc_lower_bound,
        "total_spent_uc": actual_cost_uc,
        "pct_of_cap": (
            f"{actual_cost_uc / cap_uc_lower_bound * 100:.1f}%"
            if cap_uc_lower_bound > 0 else "N/A"
        ),
        "overshoot_uc": overshoot_uc,
        "structural_undershoot_uc": 0,
        "wasted_call_cost_uc": 0,
        "wall_seconds": None,
        "workload": "lang001",
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        "byte_length_estimate": None,
        "reservation_uc": None,
        "actual_cost_uc": actual_cost_uc,
        "token_limit": token_limit,
        "token_overshoot": token_overshoot,
        "retries_total": retry_count_sum,
        "tokencap_view": tokencap_view,
        "error_repr": error_repr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-limit", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sleep", type=float, default=1.5,
        help="Inter-trial sleep (rate-limit + overload-clearance cushion)")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*72}")
    print(f"tokencap LANG-001 head-to-head benchmark (v3)")
    print(f"{'='*72}")
    print(f"  Token limit:    {args.token_limit}")
    print(f"  N trials:       {args.n_trials}")
    print(f"  Max steps:      {args.max_steps}")
    print(f"  Output:         {args.output}")
    print(f"  Model:          {ANTHROPIC_HAIKU_4_5}")
    print(f"  Strategy:       wrap() with teardown() between trials")
    print(f"  Anthropic retry: up to 5 attempts on 529 Overloaded")
    print(f"{'='*72}")

    rows = []
    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.token_limit, args.max_steps)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"  [{i+1:02d}/{args.n_trials}] HARNESS-LEVEL FAIL: "
                  f"{type(e).__name__}: {e}")
            continue
        row["wall_seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
        print(f"  [{i+1:02d}/{args.n_trials}] "
              f"outcome={row['outcome']:<35} "
              f"steps={row['agent_steps']:>2} "
              f"in={row['actual_input_tokens']:>4} "
              f"out={row['actual_output_tokens']:>4} "
              f"spent={row['total_spent_uc']:>5}uc "
              f"tok_over={row['token_overshoot']:>5} "
              f"uc_over={row['overshoot_uc']:>5} "
              f"retries={row['retries_total']}")
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
    refused = sum(1 for r in rows if "refused" in r["outcome"])
    overloaded = sum(1 for r in rows if "exhausted_retries_overloaded" in r["outcome"])
    overshoots_uc = sum(1 for r in rows if r["overshoot_uc"] > 0)
    overshoots_tok = sum(1 for r in rows if r["token_overshoot"] > 0)
    completed = sum(1 for r in rows if r["outcome"] == "max_steps_reached")
    other_errors = n - refused - completed - overloaded
    mean_steps = sum(r["agent_steps"] for r in rows) / n
    mean_spent = sum(r["total_spent_uc"] for r in rows) / n
    mean_tokens = sum(
        r["actual_input_tokens"] + r["actual_output_tokens"] for r in rows
    ) / n

    spending_trials = [r for r in rows if r["agent_steps"] > 0]
    if spending_trials:
        mean_tok_ratio = sum(
            (r["actual_input_tokens"] + r["actual_output_tokens"])
            / r["token_limit"]
            for r in spending_trials
        ) / len(spending_trials)
        mean_uc_ratio = sum(
            r["total_spent_uc"] / r["cap_uc"]
            for r in spending_trials
        ) / len(spending_trials)
    else:
        mean_tok_ratio = 0.0
        mean_uc_ratio = 0.0

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Refused (pre-call blocked):    {refused}/{n}")
    print(f"  Reached max_steps:             {completed}/{n}")
    print(f"  Anthropic overload (529):      {overloaded}/{n}")
    print(f"  Other errors:                  {other_errors}/{n}")
    print(f"  Token-overshoot trials:        {overshoots_tok}/{n}")
    print(f"  Dollar-overshoot trials:       {overshoots_uc}/{n}")
    print(f"  Mean agent_steps:              {mean_steps:.2f}")
    print(f"  Mean tokens used:              {mean_tokens:.0f}")
    print(f"  Mean spent (uc):               {mean_spent:.0f}")
    print(f"  Mean spent ($):                ${mean_spent / 1_000_000:.6f}")
    print(f"  Mean token-overshoot ratio:    {mean_tok_ratio:.2f}x")
    print(f"  Mean dollar-overshoot ratio:   {mean_uc_ratio:.2f}x")
    print(f"  ({len(spending_trials)} of {n} trials had >0 admitted calls)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()