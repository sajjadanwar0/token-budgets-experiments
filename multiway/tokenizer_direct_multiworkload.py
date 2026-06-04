import argparse
import csv
import os
import sys
import time
from anthropic import Anthropic

WORKLOADS = {
    "arg-hallucination": {
        "system": (
            "You are a function-call agent. The user will give you a task; "
            "you must call ONE of these functions: get_weather(location: str, "
            "units: 'celsius'|'fahrenheit'), get_stock_price(ticker: str), "
            "send_email(to: str, subject: str, body: str), schedule_meeting("
            "title: str, date: str, attendees: list). Respond ONLY with "
            "function(arg1=value1, arg2=value2). If you call a function with "
            "bad argument names, you will receive a validation error and must "
            "retry with the correct argument names."
        ),
        "user": (
            "Schedule a meeting for next Tuesday with john@example.com and "
            "sarah@example.com about the Q4 planning roadmap."
        ),
        "error_response": (
            "Validation error: 'when' is not a valid argument name; valid "
            "argument names are 'title', 'date', 'attendees'. Retry."
        ),
    },
    "clarification": {
        "system": (
            "You are a research assistant. The user will describe a research "
            "task. Before doing any work, ask up to 5 clarifying questions to "
            "understand the scope. Number each question 1, 2, 3, etc. After "
            "each round of clarifications, you will be asked if you need more "
            "context before starting; respond accordingly."
        ),
        "user": (
            "I need help with my dissertation about distributed systems. "
            "Can you help me?"
        ),
        "error_response": (
            "Thank you for those questions. I'll need to consult with my "
            "advisor before answering them. Please ask more clarifying "
            "questions to refine the scope further."
        ),
    },
}

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

def run_trial(trial_id, workload_name, cap_uc, max_steps=20):
    wl = WORKLOADS[workload_name]
    client = Anthropic()
    messages = [{"role": "user", "content": wl["user"]}]
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
            system=wl["system"],
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
                f"refused: predicted={predicted_uc}uc, "
                f"remaining={remaining_uc}uc, input_tokens={input_tokens}"
            )
            break

        remaining_uc -= predicted_uc

        resp, err_class, _ = call_with_retry(
            client,
            model=ANTHROPIC_HAIKU_4_5,
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=0,
            system=wl["system"],
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
        messages.append({"role": "user", "content": wl["error_response"]})

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

    mean_ct_latency = (
        total_count_tokens_latency_ms / count_tokens_calls
        if count_tokens_calls > 0 else 0
    )

    return {
        "runtime": "tokenizer_direct",
        "workload": workload_name,
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
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        "count_tokens_calls": count_tokens_calls,
        "count_tokens_total_latency_ms": round(total_count_tokens_latency_ms, 1),
        "count_tokens_mean_latency_ms": round(mean_ct_latency, 1),
        "error_repr": error_repr,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True,
        choices=list(WORKLOADS.keys()))
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
    print(f"Tokenizer-direct multi-workload benchmark")
    print(f"{'='*76}")
    print(f"  Workload:    {args.workload}")
    print(f"  Cap (uc):    {args.cap_uc}  (= ${args.cap_uc / 1_000_000:.6f})")
    print(f"  N trials:    {args.n_trials}")
    print(f"  Output:      {args.output}")
    print(f"  Model:       {ANTHROPIC_HAIKU_4_5}")
    print(f"{'='*76}")

    rows = []

    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.workload, args.cap_uc, args.max_steps)
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
        print("\nNo rows produced.")
        sys.exit(1)

    n = len(rows)
    refused = sum(1 for r in rows if r["outcome"] == "compile_time_reservation_refused")
    overshoots = sum(1 for r in rows if r["overshoot_uc"] > 0)
    spending = [r for r in rows if r["agent_steps"] > 0]

    print(f"\n{'='*76}")
    print(f"SUMMARY: {args.workload} @ cap={args.cap_uc}uc, N={n}")
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

    total_ct_calls = sum(r["count_tokens_calls"] for r in rows)

    if total_ct_calls > 0:
        mean_ct_latency = sum(r["count_tokens_total_latency_ms"] for r in rows) / total_ct_calls
        print(f"  Mean ct_lat per call:  {mean_ct_latency:.0f} ms")
    print(f"{'='*76}")

if __name__ == "__main__":
    main()