#!/usr/bin/env python3
"""
m2_run.py - standalone M2 experiment runner

Implements three pre-flight discipline adapters and writes CSV in the same
schema as multiway_compare.py's _summarise() output (runtime, outcome,
agent_steps, cap_uc, total_spent_uc, pct_of_cap, overshoot_uc,
structural_undershoot_uc, wasted_call_cost_uc). No imports from
multiway_compare.py; no edits to it. Run this script standalone.

Three adapters:

  1. token_capabilities (coarse fixed-form estimator)
     Mirrors the existing run_token_capabilities adapter in
     multiway_compare.py. Included here so the M2 CSV is self-contained
     and the coarse-vs-bytelen estimator comparison can be done within
     one CSV.

  2. token_capabilities_bytelen (NEW for M2)
     Same inline pre-flight discipline as (1) but with byte-length+2x
     margin estimator matching the Rust impl
     (token-budgets/src/estimator/default.rs). Isolates ESTIMATOR
     CHOICE when compared against (1).

  3. naive_guard (NEW for M2)
     Bare 4-line counter discipline with no per-turn estimator
     bookkeeping beyond the byte-length check. The receipt/refund
     cycle is absent. Isolates LIBRARY DISCIPLINE when compared
     against (2).

The TB-Rust comparison row comes from a separate invocation of your
tc_live_harness Rust binary. The LangGraph+AgentGuard control row
comes from a separate invocation of your existing multiway_compare.py
with --runtimes langgraph_with_guard.

The three CSVs (this script's output + Rust output + LangGraph control
output) are merged by m2_table_generator_v2.py.

Usage:
  # Smoke test with mock provider (no API key needed)
  python3 m2_run.py --provider mock --cap-uc 1500 --runs 1 \
      --output-csv m2_smoke.csv

  # Live run on gpt-4o (requires OPENAI_API_KEY)
  python3 m2_run.py --provider openai --cap-uc 1500 --runs 30 \
      --output-csv m2_gpt4o_lang001_cap1500_n30.csv

  # Live run on claude-haiku-4-5 (requires ANTHROPIC_API_KEY)
  python3 m2_run.py --provider anthropic --cap-uc 2000 --runs 30 \
      --output-csv m2_haiku_lang001_cap2000_n30.csv

Then run the existing harness for the AgentGuard control:
  python3 multiway_compare.py --provider openai --cap-uc 1500 \
      --runtimes langgraph_with_guard --runs 30 \
      --output-csv m2_gpt4o_ag_control.csv

Then merge with the table generator:
  python3 m2_table_generator_v2.py \
      --gpt4o-csv  m2_gpt4o_lang001_cap1500_n30.csv \
      --gpt4o-rust-csv m2_rust_gpt4o.csv \
      --haiku-csv m2_haiku_lang001_cap2000_n30.csv \
      --haiku-rust-csv m2_rust_haiku.csv \
      --gpt4o-cap 1500 --haiku-cap 2000 \
      --out-latex m2_table.tex --out-summary m2_summary.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Provider pricing
#
# IMPORTANT: these per-token rates are stated in micro-cents per million
# tokens. The harness uses integer division `// 1_000_000` so the result is
# an integer uc value. 1 uc = 10^-5 USD.
#
# Sync these to whatever PROVIDER_PRICING in your multiway_compare.py uses
# if you want the CSV's total_spent_uc column to match exactly. The M2
# comparison is INTERNALLY consistent regardless (all three adapters in
# this file use the same dict), but cross-CSV merging with the existing
# harness assumes the same pricing.
# =============================================================================

PROVIDER_PRICING: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model": "gpt-4o",
        # $2.50 / Mtok input, $10.00 / Mtok output
        "input_uc_per_mtok":  250_000,
        "output_uc_per_mtok": 1_000_000,
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        # $1.00 / Mtok input, $5.00 / Mtok output
        "input_uc_per_mtok":  100_000,
        "output_uc_per_mtok": 500_000,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        # $0.59 / Mtok input, $0.79 / Mtok output (Groq published rates)
        "input_uc_per_mtok":  59_000,
        "output_uc_per_mtok": 79_000,
    },
    "mock": {
        "model": "mock-llm",
        "input_uc_per_mtok":  100_000,   # use anthropic-haiku-equivalent rates
        "output_uc_per_mtok": 500_000,
    },
}


def compute_cost_uc(input_tokens: int, output_tokens: int, provider: str) -> int:
    """Returns cost in uc as an integer. Same shape as the existing harness."""
    p = PROVIDER_PRICING[provider]
    return (
            input_tokens * p["input_uc_per_mtok"]
            + output_tokens * p["output_uc_per_mtok"]
    ) // 1_000_000


# =============================================================================
# LANG-001 workload
#
# This reproduces the SQL retry loop documented in Section 2/5 of the paper.
# Model: receives a SQL task description, calls a run_sql tool, the tool
# returns a syntax error, the model retries with a different query, repeat.
#
# wl["tool_error"] and wl["tool_name"] match the names the existing harness
# uses. If your WORKLOADS["lang001"] in multiway_compare.py differs, edit
# this dict to match before running.
# =============================================================================

WORKLOADS: Dict[str, Dict[str, Any]] = {
    "lang001": {
        "name": "lang001",
        # Workload semantics: a SQL-retry loop. The model proposes a query;
        # the simulated "execution" returns a syntax-error message; the model
        # retries with a different query. The retry loop continues until the
        # cap fires.
        #
        # We use plain user/assistant turns rather than tool-calls so the
        # script works on both OpenAI and Anthropic without needing
        # tool_call_id wiring (OpenAI strictly requires that any role="tool"
        # message be preceded by a matching tool_calls field on the assistant
        # message). The growing prompt across turns is what drives cap firing;
        # the tool-vs-text format does not affect M2's separability claim.
        "retry_error": (
            "ERROR: SQL syntax error at or near the start of the query. "
            "Re-write the query with correct PostgreSQL syntax and try again."
        ),
        "system_prompt": (
            "You are a SQL assistant. When given a task, write a single "
            "PostgreSQL query. If the system returns an error, write a "
            "corrected query."
        ),
        "user_prompt": (
            "Find all employees in the engineering department earning more "
            "than $100k. Return a single PostgreSQL query."
        ),
    },
}


# =============================================================================
# Per-step accounting (CSV nesting)
# =============================================================================


def _initial_messages(wl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": wl["system_prompt"]},
        {"role": "user",   "content": wl["user_prompt"]},
    ]


def _byte_length_of_messages(messages: List[Dict[str, Any]]) -> int:
    """UTF-8 byte length of the message content payload.

    Mirrors what the Rust ByteLengthEstimator counts: the body bytes the
    LLM call would actually transmit, not the LangChain or HTTP envelope.
    """
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")).encode("utf-8"))
                else:
                    total += len(str(part).encode("utf-8"))
        else:
            total += len(str(content).encode("utf-8"))
    return total


# =============================================================================
# LLM client dispatch
#
# Three providers: openai, anthropic, mock. Plain user/assistant chat
# (no tool-calls); the retry loop drives prompt growth via appended
# user-error turns. The mock provider returns deterministic token counts
# growing with --growth per step; useful for wiring tests and CI without
# API cost.
#
# Returns a normalised dict:
#   {
#     "assistant_text": str,    # what the model said (for appending to history)
#     "input_tokens":   int,    # provider-reported input tokens
#     "output_tokens":  int,    # provider-reported output tokens
#     "self_terminated": bool,  # True if the model's output suggests it's done
#   }
# =============================================================================


def _llm_call(
        provider: str,
        messages: List[Dict[str, Any]],
        wl: Dict[str, Any],
        max_output_tokens: int = 200,
        mock_growth: int = 60,
        mock_step_idx: int = 0,
) -> Dict[str, Any]:
    if provider == "mock":
        prompt_bytes = _byte_length_of_messages(messages)
        # Bytes-as-tokens upper bound for mock accounting; growth simulates
        # the prompt expanding with each retry turn appended.
        in_tok = prompt_bytes + mock_growth * mock_step_idx
        out_tok = 40
        return {
            "assistant_text":  "SELECT * FROM employees WHERE department='engineering' AND salary > 100000;",
            "input_tokens":    in_tok,
            "output_tokens":   out_tok,
            "self_terminated": False,
        }

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=PROVIDER_PRICING["openai"]["model"],
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=0,
        )
        choice = resp.choices[0].message
        return {
            "assistant_text":  choice.content or "",
            "input_tokens":    resp.usage.prompt_tokens,
            "output_tokens":   resp.usage.completion_tokens,
            "self_terminated": False,
        }

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        anth_msgs = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=PROVIDER_PRICING["anthropic"]["model"],
            max_tokens=max_output_tokens,
            temperature=0,
            system=sys_text,
            messages=anth_msgs,
        )
        text_parts = [
            b.text for b in (resp.content or [])
            if getattr(b, "type", None) == "text"
        ]
        return {
            "assistant_text":  "".join(text_parts),
            "input_tokens":    resp.usage.input_tokens,
            "output_tokens":   resp.usage.output_tokens,
            "self_terminated": False,
        }

    raise ValueError(f"unknown provider: {provider!r}")


def _append_response(
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
        wl: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """After an assistant response, append (i) the assistant message and
    (ii) a user message with the retry-error to drive the next iteration.

    The user-error message is what makes the prompt grow turn-over-turn:
    each retry adds the assistant's previous attempt PLUS the error message,
    so the byte-length estimator's reservation grows with each step until
    the cap fires."""
    out = list(messages)
    text = response.get("assistant_text") or ""
    if not text:
        text = "(empty response)"
    out.append({"role": "assistant", "content": text})
    out.append({"role": "user",      "content": wl["retry_error"]})
    return out


# =============================================================================
# _summarise() shape matches multiway_compare.py
# =============================================================================


def _summarise(
        runtime: str,
        outcome: str,
        cap_uc: int,
        agent_steps: int,
        cumulative_uc: int,
) -> Dict[str, Any]:
    overshoot_uc = max(0, cumulative_uc - cap_uc)
    undershoot_uc = max(0, cap_uc - cumulative_uc) if "structural" in outcome else 0
    pct_of_cap = (cumulative_uc / cap_uc * 100.0) if cap_uc > 0 else 0.0
    return {
        "runtime":                  runtime,
        "outcome":                  outcome,
        "agent_steps":              agent_steps,
        "cap_uc":                   cap_uc,
        "total_spent_uc":           cumulative_uc,
        "pct_of_cap":               round(pct_of_cap, 2),
        "overshoot_uc":             overshoot_uc,
        "structural_undershoot_uc": undershoot_uc,
        "wasted_call_cost_uc":      0,
    }


# =============================================================================
# Adapter 1: token_capabilities (coarse fixed-form estimator)
# Mirrors the existing run_token_capabilities in multiway_compare.py.
# =============================================================================


def run_token_capabilities(
        provider: str,
        cap_uc: int,
        growth: int,
        recursion_limit: int,
        workload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wl = workload or WORKLOADS["lang001"]
    remaining = cap_uc
    messages = _initial_messages(wl)
    cumulative_uc = 0
    agent_steps = 0
    outcome = "completed_no_cap_hit"

    for step_idx in range(1, recursion_limit // 2 + 1):
        # Coarse fixed-form estimator (the existing one).
        agent_turns = sum(1 for m in messages if m["role"] == "assistant")
        est_input = 60 + growth * agent_turns
        est_output = 40
        est_uc = compute_cost_uc(est_input, est_output, provider)

        if est_uc > remaining:
            outcome = "compile_time_reservation_refused"
            break

        remaining -= est_uc
        resp = _llm_call(provider, messages, wl,
                         max_output_tokens=200,
                         mock_growth=growth,
                         mock_step_idx=step_idx)
        actual_uc = compute_cost_uc(
            resp["input_tokens"], resp["output_tokens"], provider,
        )
        cumulative_uc += actual_uc
        agent_steps += 1
        messages = _append_response(messages, resp, wl)
        if resp.get("self_terminated"):
            break  # workload self-terminated

    return _summarise("token_capabilities", outcome, cap_uc, agent_steps, cumulative_uc)


# =============================================================================
# Adapter 2: token_capabilities_bytelen (NEW for M2)
# Same inline discipline, byte-length+2x margin estimator (matches Rust impl).
# =============================================================================


def run_token_capabilities_bytelen(
        provider: str,
        cap_uc: int,
        growth: int,
        recursion_limit: int,
        workload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wl = workload or WORKLOADS["lang001"]
    remaining = cap_uc
    messages = _initial_messages(wl)
    cumulative_uc = 0
    agent_steps = 0
    outcome = "completed_no_cap_hit"

    anthropic_margin = 2.0 if provider == "anthropic" else 1.0
    reserved_output_tokens = 200

    for step_idx in range(1, recursion_limit // 2 + 1):
        # Byte-length+2x estimator (matches token-budgets/src/estimator/default.rs)
        prompt_bytes = _byte_length_of_messages(messages)
        est_input_tokens = int(prompt_bytes * anthropic_margin)
        est_uc = compute_cost_uc(est_input_tokens, reserved_output_tokens, provider)

        if est_uc > remaining:
            outcome = "compile_time_reservation_refused"
            break

        remaining -= est_uc
        resp = _llm_call(provider, messages, wl,
                         max_output_tokens=reserved_output_tokens,
                         mock_growth=growth,
                         mock_step_idx=step_idx)
        actual_uc = compute_cost_uc(
            resp["input_tokens"], resp["output_tokens"], provider,
        )
        cumulative_uc += actual_uc
        agent_steps += 1
        messages = _append_response(messages, resp, wl)
        if resp.get("self_terminated"):
            break

    return _summarise("token_capabilities_bytelen", outcome, cap_uc, agent_steps, cumulative_uc)


# =============================================================================
# Adapter 3: naive_guard (NEW for M2)
# Bare counter + byte-length check. NO refund discipline, NO library wrapping,
# NO receipt cycle. The discipline is: estimate, check, decrement, call.
# Difference from (2): no reservation/refund bookkeeping, no max-tracking
# beyond a single counter. If the provider returns a transient error, the
# reservation is lost (no refund).
# =============================================================================


def run_naive_guard(
        provider: str,
        cap_uc: int,
        growth: int,
        recursion_limit: int,
        workload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wl = workload or WORKLOADS["lang001"]
    remaining = cap_uc                  # the counter
    messages = _initial_messages(wl)
    cumulative_uc = 0
    agent_steps = 0
    outcome = "completed_no_cap_hit"

    anthropic_margin = 2.0 if provider == "anthropic" else 1.0
    reserved_output_tokens = 200

    for step_idx in range(1, recursion_limit // 2 + 1):
        prompt_bytes = _byte_length_of_messages(messages)
        est_input_tokens = int(prompt_bytes * anthropic_margin)
        est_uc = compute_cost_uc(est_input_tokens, reserved_output_tokens, provider)

        # ============ THE 4-LINE GUARD ============
        if est_uc > remaining:
            outcome = "compile_time_reservation_refused"
            break
        remaining -= est_uc
        # =========================================
        # NO receipt, NO refund-on-error path. If the LLM call raises,
        # the reservation is silently lost.

        try:
            resp = _llm_call(provider, messages, wl,
                             max_output_tokens=reserved_output_tokens,
                             mock_growth=growth,
                             mock_step_idx=step_idx)
        except Exception:
            # Bare counter: reservation already deducted, no refund.
            outcome = "completed_no_cap_hit"
            break

        actual_uc = compute_cost_uc(
            resp["input_tokens"], resp["output_tokens"], provider,
        )
        cumulative_uc += actual_uc
        agent_steps += 1
        messages = _append_response(messages, resp, wl)
        if resp.get("self_terminated"):
            break

    return _summarise("naive_guard", outcome, cap_uc, agent_steps, cumulative_uc)


# =============================================================================
# main()
# =============================================================================


ADAPTERS = {
    "token_capabilities":         run_token_capabilities,
    "token_capabilities_bytelen": run_token_capabilities_bytelen,
    "naive_guard":                run_naive_guard,
}

DEFAULT_RUNTIMES = ",".join(ADAPTERS.keys())


def _load_done_trials(csv_path: str) -> set:
    """Read existing CSV (if any) and return the set of (runtime, trial)
    pairs already recorded, so a resumed run skips them."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    done = set()
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rt = row.get("runtime", "")
                tr = row.get("trial", "")
                if rt and tr != "":
                    done.add((rt, int(tr)))
    except Exception as e:
        print(f"WARN: could not read existing CSV {csv_path}: {e}", file=sys.stderr)
        print(f"WARN: starting fresh; old CSV will be overwritten.", file=sys.stderr)
        return set()
    return done


def main():
    ap = argparse.ArgumentParser(
        description="M2 standalone runner: pre-flight discipline isolation experiment."
    )
    ap.add_argument("--provider",
                    choices=["mock", "openai", "anthropic"],
                    default="mock",
                    help="LLM provider. 'mock' makes no API calls (default).")
    ap.add_argument("--cap-uc", type=int, default=1500,
                    help="Budget cap in micro-cents (default 1500).")
    ap.add_argument("--growth", type=int, default=60,
                    help="Per-step input-token growth used by the coarse estimator and mock LLM (default 60).")
    ap.add_argument("--recursion-limit", type=int, default=20,
                    help="Max LLM-call attempts per run (default 20, matches multiway_compare.py).")
    ap.add_argument("--runs", type=int, default=10,
                    help="Number of trials per runtime (default 10).")
    ap.add_argument("--workload", choices=["lang001"], default="lang001",
                    help="Workload selection (only lang001 implemented in this standalone runner).")
    ap.add_argument("--runtimes", type=str, default=DEFAULT_RUNTIMES,
                    help=f"Comma-separated subset of runtimes to run (default: {DEFAULT_RUNTIMES}).")
    ap.add_argument("--output-csv", required=True,
                    help="Write per-trial rows to this CSV. Rows are flushed after every trial; "
                         "if the file exists, completed (runtime, trial) pairs are skipped on resume.")
    args = ap.parse_args()

    selected = [r.strip() for r in args.runtimes.split(",") if r.strip()]
    unknown = [r for r in selected if r not in ADAPTERS]
    if unknown:
        sys.exit(f"unknown runtime(s): {unknown}; valid: {list(ADAPTERS.keys())}")

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY env var not set; refusing to call OpenAI.")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY env var not set; refusing to call Anthropic.")

    wl = WORKLOADS[args.workload]

    # Resume support: read existing rows so we skip already-done trials.
    done = _load_done_trials(args.output_csv)
    if done:
        print(f"Resume mode: {len(done)} (runtime, trial) pairs already in "
              f"{args.output_csv} -- will skip them.", file=sys.stderr)

    # File state: open in append mode if it has content already, else write
    # header first.
    file_has_content = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0

    # Canonical field order (matches multiway_compare.py's _summarise plus the
    # metadata we add).
    FIELDNAMES = [
        "runtime", "outcome", "agent_steps", "cap_uc", "total_spent_uc",
        "pct_of_cap", "overshoot_uc", "structural_undershoot_uc",
        "wasted_call_cost_uc", "provider", "model", "workload", "trial",
    ]

    mode = "a" if file_has_content else "w"
    fh = open(args.output_csv, mode, newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not file_has_content:
        writer.writeheader()
        fh.flush()

    n_new = 0
    try:
        for runtime in selected:
            adapter = ADAPTERS[runtime]
            todo = [t for t in range(args.runs) if (runtime, t) not in done]
            skipped = args.runs - len(todo)
            msg = f"Running {runtime!r}: {len(todo)} trial(s) on {args.provider!r}, cap_uc={args.cap_uc}"
            if skipped:
                msg += f" (skipping {skipped} already done)"
            msg += "..."
            print(msg, file=sys.stderr)
            for trial in todo:
                row = adapter(
                    provider=args.provider,
                    cap_uc=args.cap_uc,
                    growth=args.growth,
                    recursion_limit=args.recursion_limit,
                    workload=wl,
                )
                row["provider"] = args.provider
                row["model"]    = PROVIDER_PRICING[args.provider]["model"]
                row["workload"] = args.workload
                row["trial"]    = trial
                writer.writerow(row)
                fh.flush()                # durable after each trial
                try:
                    os.fsync(fh.fileno()) # belt-and-braces against power loss
                except (OSError, AttributeError):
                    pass
                n_new += 1
                print(f"  trial {trial}: outcome={row['outcome']!r}, "
                      f"spent={row['total_spent_uc']} uc, overshoot={row['overshoot_uc']} uc",
                      file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\nINTERRUPTED. {n_new} new row(s) flushed to {args.output_csv}; "
              f"re-run the same command to resume.", file=sys.stderr)
        fh.close()
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR ({type(e).__name__}): {e}", file=sys.stderr)
        print(f"{n_new} new row(s) flushed to {args.output_csv} before failure; "
              f"re-run the same command to resume.", file=sys.stderr)
        fh.close()
        raise

    fh.close()
    print(f"\nDone. {n_new} new row(s) written to {args.output_csv} "
          f"({len(done) + n_new} total in file).", file=sys.stderr)


if __name__ == "__main__":
    main()