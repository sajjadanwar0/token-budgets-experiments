from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROVIDER_PRICING: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model": "gpt-4o",
        "input_uc_per_mtok":  250_000,
        "output_uc_per_mtok": 1_000_000,
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "input_uc_per_mtok":  100_000,
        "output_uc_per_mtok": 500_000,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "input_uc_per_mtok":  59_000,
        "output_uc_per_mtok": 79_000,
    },
    "mock": {
        "model": "mock-llm",
        "input_uc_per_mtok":  100_000,
        "output_uc_per_mtok": 500_000,
    },
}


def compute_cost_uc(input_tokens: int, output_tokens: int, provider: str) -> int:
    p = PROVIDER_PRICING[provider]
    return (
            input_tokens * p["input_uc_per_mtok"]
            + output_tokens * p["output_uc_per_mtok"]
    ) // 1_000_000


WORKLOADS: Dict[str, Dict[str, Any]] = {
    "lang001": {
        "name": "lang001",
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

def _initial_messages(wl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": wl["system_prompt"]},
        {"role": "user",   "content": wl["user_prompt"]},
    ]


def _byte_length_of_messages(messages: List[Dict[str, Any]]) -> int:
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
    out = list(messages)
    text = response.get("assistant_text") or ""
    if not text:
        text = "(empty response)"
    out.append({"role": "assistant", "content": text})
    out.append({"role": "user",      "content": wl["retry_error"]})
    return out

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

        if est_uc > remaining:
            outcome = "compile_time_reservation_refused"
            break
        remaining -= est_uc

        try:
            resp = _llm_call(provider, messages, wl,
                             max_output_tokens=reserved_output_tokens,
                             mock_growth=growth,
                             mock_step_idx=step_idx)
        except Exception:
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


ADAPTERS = {
    "token_capabilities":         run_token_capabilities,
    "token_capabilities_bytelen": run_token_capabilities_bytelen,
    "naive_guard":                run_naive_guard,
}

DEFAULT_RUNTIMES = ",".join(ADAPTERS.keys())


def _load_done_trials(csv_path: str) -> set:
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

    done = _load_done_trials(args.output_csv)
    if done:
        print(f"Resume mode: {len(done)} (runtime, trial) pairs already in "
              f"{args.output_csv} -- will skip them.", file=sys.stderr)

    file_has_content = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0

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
                fh.flush()
                try:
                    os.fsync(fh.fileno())
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