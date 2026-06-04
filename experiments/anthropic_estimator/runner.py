import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr)
    sys.exit(1)

MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 200
RUNS_PER_WORKLOAD = 10
WORKLOADS = ["sql_retry", "ambig_tool", "arg_hallucination"]
TOTAL_RUNS = RUNS_PER_WORKLOAD * len(WORKLOADS)

SAFETY_MARGIN = 2.0
BASE_ESTIMATOR_NAME = "ByteLength"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

SQL_RETRY_TOOLS = [{
    "name": "execute_sql",
    "description": "Execute a SQL query against the customer database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "database": {"type": "string"},
        },
        "required": ["query"],
    },
}]

SQL_RETRY_PROMPT = (
    "You have access to a `execute_sql` tool. Run a query to find all "
    "customers in the 'enterprise' tier whose contracts expire in Q1 2026. "
    "If the query fails (e.g., missing column), inspect the error message "
    "and retry with a corrected query. Continue until you succeed or "
    "determine the query is impossible."
)

AMBIG_TOOL_TOOLS = [
    {
        "name": "send_email",
        "description": "Send an email to a customer",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        },
    },
    {
        "name": "send_notification",
        "description": "Send an in-app notification",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "message": {"type": "string"},
            },
        },
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
        },
    },
]

AMBIG_TOOL_PROMPT = (
    "A customer is unhappy about a delayed shipment. Notify them about the "
    "delay. (You have multiple tools available; choose carefully.)"
)

ARG_HALLUCINATION_TOOLS = [{
    "name": "fetch_user_data",
    "description": "Fetch user profile data",
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Numeric user ID",
            },
            "fields": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["user_id"],
    },
}]

ARG_HALLUCINATION_PROMPT = (
    "Fetch the profile data for user Sarah Johnson. (Note: the tool requires "
    "a numeric user_id, but the only identifier you have is the name.)"
)

WORKLOAD_SPECS = {
    "sql_retry": (SQL_RETRY_PROMPT, SQL_RETRY_TOOLS),
    "ambig_tool": (AMBIG_TOOL_PROMPT, AMBIG_TOOL_TOOLS),
    "arg_hallucination": (ARG_HALLUCINATION_PROMPT, ARG_HALLUCINATION_TOOLS),
}

@dataclass
class RunRecord:
    run_id: str
    workload: str
    iteration: int
    prompt_bytes: int
    estimator_output_tokens: int
    actual_input_tokens: int
    actual_output_tokens: int
    count_tokens_check: int
    a1_holds: bool
    bt_ratio: float
    est_ratio: float
    safety_margin: float
    error: str = ""

def serialize_request_body(prompt_text: str, tools: list) -> str:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "tools": tools,
        "messages": [{"role": "user", "content": prompt_text}],
    }

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def anthropic_estimator_estimate(prompt_bytes: int, margin: float = SAFETY_MARGIN) -> int:
    return math.ceil(prompt_bytes * margin)

def run_one(workload: str, iteration: int, client: anthropic.Anthropic) -> RunRecord:
    prompt_text, tools = WORKLOAD_SPECS[workload]
    request_body = serialize_request_body(prompt_text, tools)
    prompt_bytes = len(request_body.encode("utf-8"))
    est_tokens = anthropic_estimator_estimate(prompt_bytes, SAFETY_MARGIN)

    try:
        ct_resp = client.messages.count_tokens(
            model=MODEL,
            tools=tools,
            messages=[{"role": "user", "content": prompt_text}],
        )
        count_tokens_value = ct_resp.input_tokens
    except Exception as e:
        return RunRecord(
            run_id=f"{workload}_{iteration:02d}",
            workload=workload,
            iteration=iteration,
            prompt_bytes=prompt_bytes,
            estimator_output_tokens=est_tokens,
            actual_input_tokens=-1,
            actual_output_tokens=-1,
            count_tokens_check=-1,
            a1_holds=False,
            bt_ratio=0.0,
            est_ratio=0.0,
            safety_margin=SAFETY_MARGIN,
            error=f"count_tokens: {type(e).__name__}: {e}",
        )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            tools=tools,
            messages=[{"role": "user", "content": prompt_text}],
        )
        actual_input = response.usage.input_tokens
        actual_output = response.usage.output_tokens
    except Exception as e:
        return RunRecord(
            run_id=f"{workload}_{iteration:02d}",
            workload=workload,
            iteration=iteration,
            prompt_bytes=prompt_bytes,
            estimator_output_tokens=est_tokens,
            actual_input_tokens=-1,
            actual_output_tokens=-1,
            count_tokens_check=count_tokens_value,
            a1_holds=False,
            bt_ratio=0.0,
            est_ratio=0.0,
            safety_margin=SAFETY_MARGIN,
            error=f"api: {type(e).__name__}: {e}",
        )

    a1_holds = est_tokens >= actual_input
    bt_ratio = prompt_bytes / actual_input if actual_input > 0 else 0.0
    est_ratio = est_tokens / actual_input if actual_input > 0 else 0.0

    return RunRecord(
        run_id=f"{workload}_{iteration:02d}",
        workload=workload,
        iteration=iteration,
        prompt_bytes=prompt_bytes,
        estimator_output_tokens=est_tokens,
        actual_input_tokens=actual_input,
        actual_output_tokens=actual_output,
        count_tokens_check=count_tokens_value,
        a1_holds=a1_holds,
        bt_ratio=bt_ratio,
        est_ratio=est_ratio,
        safety_margin=SAFETY_MARGIN,
    )

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    records: List[RunRecord] = []

    print(f"AnthropicEstimator A1 validation (CORRECTED v3)")
    print(f"  Model:          {MODEL}")
    print(f"  Base estimator: {BASE_ESTIMATOR_NAME}")
    print(f"  Safety margin:  {SAFETY_MARGIN}x")
    print(f"  Runs:           {TOTAL_RUNS} ({RUNS_PER_WORKLOAD} per workload)")
    print()
    print(f"  Estimator formula:  estimate = ceil(byte_length * {SAFETY_MARGIN})")
    print(f"  A1 holds iff:       estimate >= usage.input_tokens")
    print(f"  Expected est_ratio: 1.5-2.5x typical, with stdev > 0 (not 1.0!)")
    print()

    for workload in WORKLOADS:
        for i in range(RUNS_PER_WORKLOAD):
            print(f"  [{workload}] run {i+1}/{RUNS_PER_WORKLOAD}...",
                  end=" ", flush=True)
            t0 = time.time()
            rec = run_one(workload, i, client)
            elapsed = time.time() - t0

            if rec.error:
                print(f"ERROR ({elapsed:.1f}s): {rec.error}")
            else:
                status = "A1 holds" if rec.a1_holds else "A1 FAILS"
                print(f"{status}  est={rec.estimator_output_tokens:>4} "
                      f"actual={rec.actual_input_tokens:>4} "
                      f"ratio={rec.est_ratio:.3f}  ({elapsed:.1f}s)")

            records.append(rec)
            time.sleep(0.5)

    csv_path = OUTPUT_DIR / "runs.csv"
    with open(csv_path, "w", newline="") as f:
        if records:
            fieldnames = [f.name for f in fields(records[0])]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
    print(f"\nResults written to {csv_path}")

    valid = [r for r in records if not r.error]

    if not valid:
        print("\n!!! No valid runs; check API key and connectivity.")
        return

    a1_pass = sum(1 for r in valid if r.a1_holds)
    est_ratios = [r.est_ratio for r in valid]
    bt_ratios = [r.bt_ratio for r in valid]

    def stats(xs):
        n = len(xs)
        mean = sum(xs) / n
        sd = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
        return mean, sd, min(xs), max(xs)

    est_mean, est_sd, est_min, est_max = stats(est_ratios)
    bt_mean, bt_sd, bt_min, bt_max = stats(bt_ratios)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total runs:          {len(records)}")
    print(f"Valid runs:          {len(valid)}")
    print(f"A1 holds:            {a1_pass}/{len(valid)} "
          f"({100*a1_pass/len(valid):.1f}%)")
    print()
    print(f"est_ratio (margin {SAFETY_MARGIN}x): "
          f"mean={est_mean:.4f}  sd={est_sd:.4f}  "
          f"range=[{est_min:.4f}, {est_max:.4f}]")
    print(f"bt_ratio  (no margin):  "
          f"mean={bt_mean:.4f}  sd={bt_sd:.4f}  "
          f"range=[{bt_min:.4f}, {bt_max:.4f}]")
    print()


    print("SANITY CHECKS")
    print("-" * 60)

    if est_sd == 0.0:
        print("WARN: est_ratio stdev is exactly 0. Harness may still be wrong.")
    else:
        print(f"OK:   est_ratio has non-zero variance (stdev={est_sd:.4f}).")

    matches = sum(1 for r in valid if r.estimator_output_tokens == r.actual_input_tokens)

    if matches == len(valid):
        print(f"WARN: estimator_output equals actual_input on every row. "
              f"Likely still computing count_tokens.")
    else:
        print(f"OK:   estimator_output differs from actual_input on "
              f"{len(valid) - matches}/{len(valid)} rows.")

    ct_matches = sum(1 for r in valid if r.count_tokens_check == r.actual_input_tokens)
    print(f"DIAG: count_tokens matched usage.input_tokens on "
          f"{ct_matches}/{len(valid)} rows "
          f"(Anthropic's own tokenizer determinism).")

    print()
    print("PER-WORKLOAD BREAKDOWN")
    print("-" * 60)

    for workload in WORKLOADS:
        wl_records = [r for r in valid if r.workload == workload]
        if not wl_records:
            continue
        wl_pass = sum(1 for r in wl_records if r.a1_holds)
        wl_ratios = [r.est_ratio for r in wl_records]
        wl_mean = sum(wl_ratios) / len(wl_ratios)
        wl_min = min(wl_ratios)
        wl_max = max(wl_ratios)
        print(f"  {workload:>20}: A1 {wl_pass}/{len(wl_records)}, "
              f"ratio mean={wl_mean:.3f} range=[{wl_min:.3f}, {wl_max:.3f}]")

    print()


if __name__ == "__main__":
    main()