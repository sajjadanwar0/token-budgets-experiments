#!/usr/bin/env python3
"""
AnthropicEstimator A1 validation harness — FIXED v2.

CRITICAL FIX vs v1:
  v1's estimate_via_anthropic_estimator() called count_tokens with only the
  user message, NOT the tools. The actual API call to messages.create()
  includes tools, so the bill counts tool-description tokens. The result
  was a massive false-failure of A1 (est_ratio ~0.078, meaning the
  estimator was counting only ~8% of what Anthropic actually billed).

  This v2 passes the FULL request payload to count_tokens — including
  tools — so the estimator and the bill see the same input.

Methodology unchanged: 30 runs (3 workloads × 10 each), capture
(estimator_output, actual_input_tokens) per run, compute A1 hold rate.
"""

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr)
    sys.exit(1)

# === Configuration ===

MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 200
RUNS_PER_WORKLOAD = 10
WORKLOADS = ["sql_retry", "ambig_tool", "arg_hallucination"]
TOTAL_RUNS = RUNS_PER_WORKLOAD * len(WORKLOADS)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# === Workload definitions ===

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
    a1_holds: bool
    bt_ratio: float
    est_ratio: float
    error: str = ""


def estimate_full_request(text: str, tools: list, client) -> int:
    """
    FIXED v2: pass tools to count_tokens so the estimator sees the same
    input the actual API call will bill for.

    This mirrors what an artifact-side AnthropicEstimator should do:
    serialize the FULL request body (messages + tools + system) and ask
    Anthropic's tokenizer how many tokens it represents.
    """
    response = client.messages.count_tokens(
        model=MODEL,
        tools=tools,                                      # ← THE FIX
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens


def run_one(workload: str, iteration: int, client: anthropic.Anthropic) -> RunRecord:
    prompt_text, tools = WORKLOAD_SPECS[workload]

    # Byte-length is computed against the same payload serialization the
    # paper's ByteLengthEstimator does (full JSON of the request body).
    request_body = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "tools": tools,
        "messages": [{"role": "user", "content": prompt_text}],
    })
    prompt_bytes = len(request_body.encode("utf-8"))

    # FIXED: pass tools to count_tokens
    try:
        est_tokens = estimate_full_request(prompt_text, tools, client)
    except Exception as e:
        return RunRecord(
            run_id=f"{workload}_{iteration:02d}",
            workload=workload,
            iteration=iteration,
            prompt_bytes=prompt_bytes,
            estimator_output_tokens=-1,
            actual_input_tokens=-1,
            actual_output_tokens=-1,
            a1_holds=False,
            bt_ratio=0.0,
            est_ratio=0.0,
            error=f"estimator: {type(e).__name__}: {e}",
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
            a1_holds=False,
            bt_ratio=0.0,
            est_ratio=0.0,
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
        a1_holds=a1_holds,
        bt_ratio=bt_ratio,
        est_ratio=est_ratio,
    )


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    records: List[RunRecord] = []

    print(f"AnthropicEstimator A1 validation (FIXED v2)")
    print(f"  Model: {MODEL}, max_output_tokens: {MAX_OUTPUT_TOKENS}")
    print(f"  Runs:  {TOTAL_RUNS} ({RUNS_PER_WORKLOAD} per workload)")
    print(f"  Key fix vs v1: count_tokens now receives tools (matching the")
    print(f"                 messages.create payload that gets billed)")
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
                status = "✓ A1 holds" if rec.a1_holds else "✗ A1 FAILS"
                print(f"{status} (est={rec.estimator_output_tokens}, "
                      f"actual={rec.actual_input_tokens}, "
                      f"ratio={rec.est_ratio:.3f}) [{elapsed:.1f}s]")

            records.append(rec)
            time.sleep(0.5)

    csv_path = OUTPUT_DIR / "runs.csv"
    with open(csv_path, "w", newline="") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
    print(f"\nResults written to {csv_path}")

    valid = [r for r in records if not r.error]
    a1_pass = sum(1 for r in valid if r.a1_holds)
    print(f"\n=== QUICK SUMMARY ===")
    print(f"Total runs:    {len(records)}")
    print(f"Valid runs:    {len(valid)}")
    print(f"A1 holds:      {a1_pass}/{len(valid)}")
    if valid:
        avg_est_ratio = sum(r.est_ratio for r in valid) / len(valid)
        print(f"Avg est_ratio: {avg_est_ratio:.4f} (expect ≥ 1.00 if fix worked)")
    print()
    print("Run `python3 analyze.py results/runs.csv` for full report.")


if __name__ == "__main__":
    main()