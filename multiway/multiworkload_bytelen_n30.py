import argparse
import csv
import json
import math
import os
import sys
import time

from anthropic import Anthropic

WORKLOADS = {
    "lang001": {
        "system": (
            "You are a SQL agent. The user will give you a task. You must write a "
            "SQL query to accomplish it. After you write each query, you will be "
            "told whether it succeeded or failed. If it failed, write a corrected "
            "query that addresses the specific error reported."
        ),
        "user": (
            "Give me the average salary by department from the 'employees' table. "
            "Group by department, order by avg salary desc."
        ),
        "fake_feedback": (
            "Error: SQL syntax error near 'AVG' on line 1. The query failed to "
            "parse. Please rewrite the query with correct SQL syntax."
        ),
        "self_terminating": False,
    },
    "clarification": {
        "system": (
            "You are a helpful task assistant. When the user's request is unclear, "
            "ask exactly one concise clarifying question and then stop. Do not "
            "speculate or proceed without an answer."
        ),
        "user": (
            "I need to book a flight for next month."
        ),
        "fake_feedback": None,
        "self_terminating": True,
    },
    "arg_hallucination": {
        "system": (
            "You are an agent that calls tools. You have access to a tool called "
            "search_users(email: str). You must call this tool to find user "
            "information. After each call, you will be told the result."
        ),
        "user": (
            "Find the user with email alice@example.com."
        ),
        "fake_feedback": (
            "Error: tool call failed. The argument format was wrong. Re-try the "
            "search with a corrected argument format."
        ),
        "self_terminating": False,
    },
}

ANTHROPIC_HAIKU_4_5 = "claude-haiku-4-5-20251001"
PRICING_UC_PER_TOKEN = {"input": 1, "output": 5}
MAX_COMPLETION_TOKENS = 200
SAFETY_MARGIN = 2.0


def serialize_request_body(model, max_tokens, system, messages):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def predict_cost_bytelen(system, messages, max_completion_tokens, margin):
    body = serialize_request_body(
        ANTHROPIC_HAIKU_4_5, max_completion_tokens, system, messages,
    )
    input_byte_len = len(body.encode("utf-8"))
    predicted_input_uc = math.ceil(input_byte_len * margin)
    predicted_output_uc = max_completion_tokens * PRICING_UC_PER_TOKEN["output"]
    return predicted_input_uc + predicted_output_uc, input_byte_len


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


def run_trial(trial_id, cap_uc, workload_name, max_steps=20):
    wl = WORKLOADS[workload_name]
    client = Anthropic()
    messages = [{"role": "user", "content": wl["user"]}]

    remaining_uc = cap_uc
    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    outcome = "max_steps_reached"
    error_repr = ""

    for step in range(max_steps):
        predicted_uc, byte_len = predict_cost_bytelen(
            wl["system"], messages, MAX_COMPLETION_TOKENS, SAFETY_MARGIN,
        )

        if predicted_uc > remaining_uc:
            outcome = "compile_time_reservation_refused"
            error_repr = (
                f"refused: predicted={predicted_uc}uc, remaining={remaining_uc}uc, "
                f"byte_length={byte_len}"
            )
            break

        remaining_uc -= predicted_uc

        resp, err_class = call_with_retry(
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
        actual_cost = (actual_input * PRICING_UC_PER_TOKEN["input"]
                       + actual_output * PRICING_UC_PER_TOKEN["output"])
        refund = predicted_uc - actual_cost
        if refund > 0:
            remaining_uc += refund

        steps += 1
        total_input_tokens += actual_input
        total_output_tokens += actual_output

        assistant_text = resp.content[0].text if resp.content else ""
        messages.append({"role": "assistant", "content": assistant_text})

        if wl["self_terminating"]:
            outcome = "completed_no_cap_hit"
            break

        messages.append({"role": "user", "content": wl["fake_feedback"]})

    actual_total_cost_uc = (
        total_input_tokens * PRICING_UC_PER_TOKEN["input"]
        + total_output_tokens * PRICING_UC_PER_TOKEN["output"]
    )
    overshoot_uc = max(0, actual_total_cost_uc - cap_uc)

    return {
        "runtime": "tb_bytelen_python",
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
        "workload": workload_name,
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        "error_repr": error_repr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True, choices=list(WORKLOADS.keys()))
    parser.add_argument("--cap-uc", type=int, default=2000)
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
    print(f"TB-default (byte-length + 2.0x margin) multi-workload sweep")
    print(f"{'='*76}")
    print(f"  Workload:   {args.workload}")
    print(f"  Cap (uc):   {args.cap_uc}")
    print(f"  N trials:   {args.n_trials}")
    print(f"  Self-terminating: {WORKLOADS[args.workload]['self_terminating']}")
    print(f"  Output:     {args.output}")
    print(f"{'='*76}")

    rows = []
    for i in range(args.n_trials):
        start = time.monotonic()
        try:
            row = run_trial(i, args.cap_uc, args.workload, args.max_steps)
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
              f"over={row['overshoot_uc']:>4}uc")
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
    overshoots = sum(1 for r in rows if r["overshoot_uc"] > 0)
    mean_spend = sum(r["total_spent_uc"] for r in rows) / n
    mean_steps = sum(r["agent_steps"] for r in rows) / n

    print(f"\n{'='*76}")
    print(f"SUMMARY: {args.workload} @ cap={args.cap_uc}uc, N={n}")
    print(f"{'='*76}")
    print(f"  Overshoot:        {overshoots}/{n}")
    print(f"  Mean steps:       {mean_steps:.2f}")
    print(f"  Mean spend (uc):  {mean_spend:.0f}")
    print(f"  % of cap:         {mean_spend/args.cap_uc*100:.1f}%")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()