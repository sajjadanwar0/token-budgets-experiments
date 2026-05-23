#!/usr/bin/env python3
"""
anthropic_multiworkload_n30.py

Runs LANG-001 retry-loop reproduction on Anthropic Sonnet under
the Token Budgets discipline for two additional workloads at N=30:
  - arg-hallucination (smolagents-style mid-action argument fabrication)
  - clarification (CrewAI-style infinite clarification loop)

Existing artefacts:
  multiway/sweep_results/tokenizer_direct_arg-hallucination_cap{2000,5000}_n10.csv
  multiway/sweep_results/tokenizer_direct_clarification_cap{2000,5000}_n10.csv

These were at N=10. This script bumps to N=30 on Anthropic Sonnet
(claude-sonnet-4-5-20250929) for direct comparison against the
N=30 baseline already in the paper (claude_sonnet_lang001_n30_full.csv).

Output files (committed to multiway/sweep_results/):
  tb_sonnet_arg-hallucination_cap{2000,5000}_n30.csv
  tb_sonnet_clarification_cap{2000,5000}_n30.csv

Cost estimate: 4 caps x 30 runs x mean 4 calls/run x mean 600 input
+ 200 output tokens at sonnet pricing ($3/M input, $15/M output)
  = 120 runs * 4 calls/run * (600 * 3 + 200 * 15) / 1e6
  = 120 * 4 * (1800 + 3000) / 1e6
  = ~$2.30
Budget liberally for $5 to absorb retries.

Requirements:
  pip install anthropic tiktoken
  export ANTHROPIC_API_KEY=...

Usage:
  python anthropic_multiworkload_n30.py
"""

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr)
    sys.exit(1)

# =========================================================================
# Configuration
# =========================================================================

MODEL = "claude-sonnet-4-5-20250929"
PROVIDER = "anthropic"
INPUT_RATE_UC_PER_TOK = 3       # $3 / Mtok input
OUTPUT_RATE_UC_PER_TOK = 15     # $15 / Mtok output
SAFETY_MARGIN = 2.0             # AnthropicEstimator default
N_TRIALS = 30
CAPS = [2000, 5000]             # token caps to sweep
WORKLOADS = ["arg-hallucination", "clarification"]
MAX_STEPS_PER_TRIAL = 10
OVERLOAD_BACKOFF_S = [2, 5, 10, 20, 30]
OUTPUT_DIR = Path("multiway/sweep_results")

# =========================================================================
# Workload prompts (derived from the catalog cases referenced in paper)
# =========================================================================

WORKLOAD_PROMPTS = {
    "arg-hallucination": {
        "system": (
            "You are a tool-calling agent. The user will ask you to plan "
            "a multi-step task. You must invoke tools to gather information. "
            "If a tool argument is unclear, fabricate a plausible value and "
            "proceed; do not stop to clarify. Continue until the task "
            "appears complete."
        ),
        "user": (
            "Search the web for recent papers on quantum error correction "
            "from 2024 onwards. For each result, fetch the abstract, "
            "summarise it in 50 words, then propose a follow-up question. "
            "Output 5 candidate papers minimum."
        ),
    },
    "clarification": {
        "system": (
            "You are a planning agent. The user will give you a task. Before "
            "acting, you must clarify any ambiguity by asking the user "
            "exactly one clarifying question at a time. If the user's "
            "response is itself ambiguous, ask another clarifying question. "
            "Do not act on the task until you are fully certain."
        ),
        "user": (
            "I want to write a report. Can you help?"
        ),
    },
}

# Simulated user response for the clarification loop (forces multi-turn)
CLARIFICATION_USER_REPLIES = [
    "It's about something I'm working on.",
    "Yes, technical.",
    "Maybe medium length.",
    "I think both audiences.",
    "Let's say next week.",
    "OK fine, just start.",
]

# =========================================================================
# Token Budget runtime port (matches the Rust Budget<MAX> discipline)
# =========================================================================

@dataclass
class Budget:
    """Affine budget capability. Returns new Budget after spend."""
    initial_uc: int
    max_uc: int
    _consumed: bool = field(default=False)

    def spend(self, amount_uc: int):
        if self._consumed:
            raise RuntimeError("Budget already consumed")
        if amount_uc > self.initial_uc:
            raise RuntimeError(
                f"Insufficient: requested {amount_uc} uc, only {self.initial_uc} available"
            )
        self._consumed = True
        return Budget(initial_uc=self.initial_uc - amount_uc, max_uc=self.max_uc)

    def micro_cents(self):
        return self.initial_uc


def estimate_uc(prompt_chars: int, max_output_tokens: int) -> int:
    """AnthropicEstimator with 2.0x safety margin.
    Reserves byte_length * 2.0 * input_rate + max_output_tokens * output_rate."""
    estimated_input_tokens = int(prompt_chars * SAFETY_MARGIN)
    reserved_input_uc = estimated_input_tokens * INPUT_RATE_UC_PER_TOK
    reserved_output_uc = max_output_tokens * OUTPUT_RATE_UC_PER_TOK
    return reserved_input_uc + reserved_output_uc


# =========================================================================
# Harness
# =========================================================================

def call_anthropic_with_retry(client, messages, system, max_tokens):
    """Call API with 529 overload retry. Returns (response, retries_used)."""
    for attempt, backoff in enumerate([0] + OVERLOAD_BACKOFF_S):
        if backoff > 0:
            time.sleep(backoff)
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return resp, attempt
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < len(OVERLOAD_BACKOFF_S):
                continue
            raise
    raise RuntimeError("Exhausted retries")


def run_trial(client, workload: str, cap_tokens: int, trial_id: int) -> dict:
    """One trial of the retry-loop reproduction.

    Returns row for CSV with columns:
      workload, cap_tokens, cap_uc, trial_id, steps, total_input_tokens,
      total_output_tokens, total_billed_uc, total_reserved_uc,
      overshoot, refused_at_step, refused_reason, retries_total
    """
    cap_uc = cap_tokens * INPUT_RATE_UC_PER_TOK * SAFETY_MARGIN  # convert token-cap to uc with margin
    budget = Budget(initial_uc=int(cap_uc), max_uc=int(cap_uc))

    spec = WORKLOAD_PROMPTS[workload]
    messages = [{"role": "user", "content": spec["user"]}]
    total_input_tokens = 0
    total_output_tokens = 0
    total_billed_uc = 0
    total_reserved_uc = 0
    steps = 0
    refused_at_step = None
    refused_reason = None
    retries_total = 0

    max_output_tokens = 500  # per-call cap

    for step in range(MAX_STEPS_PER_TRIAL):
        # Build the prompt (concatenated message history for byte-length estimate)
        prompt_chars = sum(
            len(m["content"]) if isinstance(m["content"], str) else 0
            for m in messages
        ) + len(spec["system"])

        # Pre-flight reservation
        required_uc = estimate_uc(prompt_chars, max_output_tokens)
        try:
            budget = budget.spend(required_uc)
        except RuntimeError as e:
            refused_at_step = step
            refused_reason = str(e)
            break
        total_reserved_uc += required_uc

        # Make the API call
        try:
            resp, retries = call_anthropic_with_retry(
                client, messages, spec["system"], max_output_tokens
            )
            retries_total += retries
        except Exception as e:
            refused_at_step = step
            refused_reason = f"api_error: {e}"
            break

        # Track actual billed cost
        actual_input_uc = resp.usage.input_tokens * INPUT_RATE_UC_PER_TOK
        actual_output_uc = resp.usage.output_tokens * OUTPUT_RATE_UC_PER_TOK
        actual_call_uc = actual_input_uc + actual_output_uc
        total_input_tokens += resp.usage.input_tokens
        total_output_tokens += resp.usage.output_tokens
        total_billed_uc += actual_call_uc
        steps += 1

        # Refund unused reservation
        # (in Rust this is the Refund / spend_with_receipt pattern;
        # in this Python harness we reconcile by adding back to the budget
        # the difference between reserved and actual)
        refund = required_uc - actual_call_uc
        if refund > 0:
            budget = Budget(
                initial_uc=budget.initial_uc + refund,
                max_uc=budget.max_uc,
            )

        # Append assistant response to history
        assistant_text = "".join(
            b.text for b in resp.content if hasattr(b, "text")
        )
        messages.append({"role": "assistant", "content": assistant_text})

        # Workload-specific continuation logic
        if workload == "clarification":
            # If this is a question, simulate user reply
            if "?" in assistant_text:
                reply_idx = min(step, len(CLARIFICATION_USER_REPLIES) - 1)
                messages.append({
                    "role": "user",
                    "content": CLARIFICATION_USER_REPLIES[reply_idx],
                })
            else:
                # Agent stopped asking; we're done
                break
        elif workload == "arg-hallucination":
            # Always continue; the prompt forces multi-step exploration
            messages.append({
                "role": "user",
                "content": "Continue with the next step. Be specific.",
            })

    overshoot = 1 if total_billed_uc > cap_uc else 0

    return {
        "workload": workload,
        "cap_tokens": cap_tokens,
        "cap_uc": int(cap_uc),
        "trial_id": trial_id,
        "steps": steps,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_billed_uc": total_billed_uc,
        "total_reserved_uc": total_reserved_uc,
        "overshoot": overshoot,
        "refused_at_step": refused_at_step if refused_at_step is not None else "",
        "refused_reason": refused_reason or "",
        "retries_total": retries_total,
    }


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: export ANTHROPIC_API_KEY=...", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    for workload in WORKLOADS:
        for cap_tokens in CAPS:
            out_path = OUTPUT_DIR / f"tb_sonnet_{workload}_cap{cap_tokens}_n30.csv"
            print(f"=== {workload} cap={cap_tokens} N={N_TRIALS} ===")
            print(f"    -> {out_path}")

            with open(out_path, "w", newline="") as f:
                fieldnames = [
                    "workload", "cap_tokens", "cap_uc", "trial_id", "steps",
                    "total_input_tokens", "total_output_tokens",
                    "total_billed_uc", "total_reserved_uc",
                    "overshoot", "refused_at_step", "refused_reason",
                    "retries_total",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for trial in range(N_TRIALS):
                    print(f"    trial {trial+1}/{N_TRIALS}", end=" ", flush=True)
                    row = run_trial(client, workload, cap_tokens, trial)
                    writer.writerow(row)
                    f.flush()
                    indicator = "OVER" if row["overshoot"] else "ok"
                    refused = (
                        f" refused@{row['refused_at_step']}"
                        if row["refused_at_step"] != "" else ""
                    )
                    print(f"steps={row['steps']} billed={row['total_billed_uc']} [{indicator}]{refused}")

            # Quick summary
            with open(out_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                overshoots = sum(int(r["overshoot"]) for r in rows)
                mean_steps = sum(int(r["steps"]) for r in rows) / len(rows)
                mean_billed = sum(int(r["total_billed_uc"]) for r in rows) / len(rows)
                print(f"    SUMMARY: {overshoots}/{N_TRIALS} overshoot, "
                      f"mean steps={mean_steps:.2f}, mean billed={mean_billed:.0f} uc")
            print()

    print("Done. CSVs are in multiway/sweep_results/")


if __name__ == "__main__":
    main()