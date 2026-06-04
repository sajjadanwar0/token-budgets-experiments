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

def predict_cost_uc(client, messages, system, max_completion_tokens):
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

def run_trial(trial_id, cap_uc, max_steps=20):
    client = Anthropic()
    messages = [{"role": "user", "content": LANG_001_USER}]
    remaining_uc = cap_uc
    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    outcome = "max_steps_reached"
    error_repr = ""
    count_tokens_calls = 0
    total_count_tokens_latency_ms = 0.0

    for step in range(max_steps):
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