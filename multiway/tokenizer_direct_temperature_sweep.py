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


def run_trial(trial_id, cap_uc, temperature, max_steps=20):
    client = Anthropic()
    messages = [{"role": "user", "content": LANG_001_USER}]

    remaining_uc = cap_uc
    total_input_tokens = 0
    total_output_tokens = 0
    output_tokens_per_step = []
    steps = 0
    outcome = "max_steps_reached"
    error_repr = ""

    for step in range(max_steps):
        predicted_uc, input_tokens, latency_ms, err = predict_cost_uc(
            client,
            messages=messages,
            system=LANG_001_SYSTEM,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

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
            temperature=temperature,
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
        output_tokens_per_step.append(actual_output)

        assistant_text = resp.content[0].text if resp.content else ""
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": LANG_001_FAKE_ERROR})

    actual_total_cost_uc = (
        total_input_tokens * PRICING_UC_PER_TOKEN["input"]
        + total_output_tokens * PRICING_UC_PER_TOKEN["output"]
    )
    overshoot_uc = max(0, actual_total_cost_uc - cap_uc)

    return {
        "runtime": "tokenizer_direct",
        "temperature": temperature,
        "run_id": f"trial_{trial_id}",
        "provider": "anthropic",
        "outcome": outcome,
        "agent_steps": steps,
        "cap_uc": cap_uc,
        "total_spent_uc": actual_total_cost_uc,
        "remaining_uc": remaining_uc,
        "pct_of_cap": f"{actual_total_cost_uc / cap_uc * 100:.1f}%" if cap_uc else "N/A",
        "overshoot_uc": overshoot_uc,
        "wall_seconds": None,
        "workload": "lang001",
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        "output_tokens_per_step": ",".join(str(o) for o in output_tokens_per_step),
        "error_repr": error_repr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cap-uc", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=10)
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
    print(f"Tokenizer-direct temperature sweep")
    print(f"{'='*76}")
    print(f"  T:           {args.temperature}")
    print(f"  Cap (uc):    {args.cap_uc}")
    print(f"  N trials:    {args.n_trials}")
    print(f"  Output:      {args.output}")
    print(f"{'='*76}")

    rows = []
    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.cap_uc, args.temperature, args.max_steps)
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"  [{i+1:02d}/{args.n_trials}] HARNESS FAIL: {type(e).__name__}: {e}")
            continue
        row["wall_seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
        print(f"  [{i+1:02d}/{args.n_trials}] "
              f"outcome={row['outcome'][:30]:<30} "
              f"steps={row['agent_steps']:>2} "
              f"spent={row['total_spent_uc']:>5}uc "
              f"over={row['overshoot_uc']:>4}uc "
              f"out_toks={row['output_tokens_per_step']}")
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
    overshoots = sum(1 for r in rows if r["overshoot_uc"] > 0)
    spending = [r for r in rows if r["agent_steps"] > 0]
    if spending:
        all_outputs = []
        for r in rows:
            if r["output_tokens_per_step"]:
                all_outputs.extend(int(x) for x in r["output_tokens_per_step"].split(","))

    print(f"\n{'='*76}")
    print(f"SUMMARY: T={args.temperature}, cap={args.cap_uc}uc, N={n}")
    print(f"{'='*76}")
    print(f"  Overshoot:        {overshoots}/{n}")
    if spending:
        mean_steps = sum(r["agent_steps"] for r in spending) / len(spending)
        mean_spent = sum(r["total_spent_uc"] for r in spending) / len(spending)
        print(f"  Spending trials:  {len(spending)}/{n}")
        print(f"  Mean steps:       {mean_steps:.2f}")
        print(f"  Mean spent (uc):  {mean_spent:.0f}")
        if all_outputs:
            from statistics import mean, stdev
            mean_o = mean(all_outputs)
            sd_o = stdev(all_outputs) if len(all_outputs) > 1 else 0
            print(f"  Output tokens:    n={len(all_outputs)}, "
                  f"mean={mean_o:.0f}, sd={sd_o:.0f}, "
                  f"min={min(all_outputs)}, max={max(all_outputs)}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()