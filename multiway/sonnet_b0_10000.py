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

MODEL = "claude-sonnet-4-5-20250929"
INPUT_RATE_UC_PER_TOK = 3
OUTPUT_RATE_UC_PER_TOK = 15
SAFETY_MARGIN = 2.0
N_TRIALS = 30
CAP_TOKENS = 10000
MAX_STEPS_PER_TRIAL = 12
OUTPUT_DIR = Path("multiway/sweep_results")
OVERLOAD_BACKOFF_S = [2, 5, 10, 20, 30]

LANG001_SYSTEM = (
    "You are a tool-calling agent. Search for information to answer "
    "the user's question. If you encounter an error, retry with a "
    "different approach. Continue until you have a complete answer."
)
LANG001_USER = (
    "Find the GDP of Italy, France, and Germany for 2023. "
    "Compare them and identify which is highest. "
    "Show your reasoning step by step."
)


@dataclass
class Budget:
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


def estimate_uc(prompt_chars: int, max_output_tokens: int) -> int:
    est_input = int(prompt_chars * SAFETY_MARGIN)
    return est_input * INPUT_RATE_UC_PER_TOK + max_output_tokens * OUTPUT_RATE_UC_PER_TOK


def call_with_retry(client, messages, system, max_tokens):
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


def run_trial(client, trial_id: int) -> dict:
    cap_uc = CAP_TOKENS * INPUT_RATE_UC_PER_TOK * SAFETY_MARGIN
    budget = Budget(initial_uc=int(cap_uc), max_uc=int(cap_uc))
    messages = [{"role": "user", "content": LANG001_USER}]
    total_in_tok = 0
    total_out_tok = 0
    total_billed_uc = 0
    total_reserved_uc = 0
    steps = 0
    refused_at_step = ""
    refused_reason = ""
    retries_total = 0

    for step in range(MAX_STEPS_PER_TRIAL):
        prompt_chars = sum(
            len(m["content"]) for m in messages if isinstance(m.get("content"), str)
        ) + len(LANG001_SYSTEM)
        required_uc = estimate_uc(prompt_chars, max_output_tokens=500)
        try:
            budget = budget.spend(required_uc)
        except RuntimeError as e:
            refused_at_step = step
            refused_reason = str(e)
            break
        total_reserved_uc += required_uc

        try:
            resp, retries = call_with_retry(client, messages, LANG001_SYSTEM, 500)
            retries_total += retries
        except Exception as e:
            refused_at_step = step
            refused_reason = f"api_error: {e}"
            break

        in_uc = resp.usage.input_tokens * INPUT_RATE_UC_PER_TOK
        out_uc = resp.usage.output_tokens * OUTPUT_RATE_UC_PER_TOK
        actual_uc = in_uc + out_uc
        total_in_tok += resp.usage.input_tokens
        total_out_tok += resp.usage.output_tokens
        total_billed_uc += actual_uc
        steps += 1

        refund = required_uc - actual_uc
        if refund > 0:
            budget = Budget(
                initial_uc=budget.initial_uc + refund,
                max_uc=budget.max_uc,
            )

        assistant_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        messages.append({"role": "assistant", "content": assistant_text})

        # Continue the loop with a generic continuation
        messages.append({
            "role": "user",
            "content": "Continue if you have more steps; otherwise summarise.",
        })

        if len(assistant_text) < 100 and step > 0:
            break

    overshoot = 1 if total_billed_uc > cap_uc else 0

    return {
        "cap_tokens": CAP_TOKENS,
        "cap_uc": int(cap_uc),
        "trial_id": trial_id,
        "steps": steps,
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "total_billed_uc": total_billed_uc,
        "total_reserved_uc": total_reserved_uc,
        "overshoot": overshoot,
        "refused_at_step": refused_at_step,
        "refused_reason": refused_reason,
        "retries_total": retries_total,
    }


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: export ANTHROPIC_API_KEY=...", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"tb_sonnet_lang001_cap{CAP_TOKENS}_n30.csv"
    client = anthropic.Anthropic()
    print(f"=== sonnet LANG-001 cap={CAP_TOKENS} N={N_TRIALS} ===")
    print(f"    -> {out_path}")

    fieldnames = [
        "cap_tokens", "cap_uc", "trial_id", "steps",
        "total_input_tokens", "total_output_tokens",
        "total_billed_uc", "total_reserved_uc",
        "overshoot", "refused_at_step", "refused_reason",
        "retries_total",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in range(N_TRIALS):
            print(f"    trial {trial+1}/{N_TRIALS}", end=" ", flush=True)
            row = run_trial(client, trial)
            writer.writerow(row)
            f.flush()
            ind = "OVER" if row["overshoot"] else "ok"
            ref = f" refused@{row['refused_at_step']}" if row["refused_at_step"] != "" else ""
            print(f"steps={row['steps']} billed={row['total_billed_uc']}uc [{ind}]{ref}")

    # Summary
    with open(out_path, "r") as f:
        rows = list(csv.DictReader(f))
        overshoots = sum(int(r["overshoot"]) for r in rows)
        mean_steps = sum(int(r["steps"]) for r in rows) / len(rows)
        mean_billed = sum(int(r["total_billed_uc"]) for r in rows) / len(rows)
        print()
        print(f"SUMMARY: {overshoots}/{N_TRIALS} overshoot, "
              f"mean steps={mean_steps:.2f}, mean billed={mean_billed:.0f} uc")


if __name__ == "__main__":
    main()