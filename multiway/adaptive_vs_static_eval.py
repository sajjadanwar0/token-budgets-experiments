import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr)
    sys.exit(1)

MODEL = "claude-sonnet-4-5-20250929"
INPUT_RATE_UC_PER_TOK = 3
OUTPUT_RATE_UC_PER_TOK = 15
MAX_OUTPUT_TOKENS = 200
STATIC_MARGIN = 2.0
ADAPTIVE_EPSILON = 0.10
OUTPUT_DIR = Path("multiway/sweep_results")
OVERLOAD_BACKOFF_S = [2, 5, 10, 20, 30]

SIMPLE_QA = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is 17 times 23?",
    "Who wrote Hamlet?",
    "What year did the moon landing happen?",
    "Name three primary colours.",
    "What is the speed of light?",
    "How many continents are there?",
    "Define entropy.",
    "What is the largest planet?",
    "Who invented the telephone?",
    "What is the chemical symbol for gold?",
    "Spell 'Mississippi'.",
    "What is a prime number?",
    "Translate 'hello' to Spanish.",
    "What is gravity?",
    "List the planets in our solar system.",
    "What is Pi to four decimal places?",
    "Define DNA.",
    "What is photosynthesis?",
]

TOOL_CALL_PROMPTS = [
    "Plan a trip to Tokyo for 5 days with daily themes, restaurants, and budget.",
    "List 10 packages for a Python data-science project with one-line justifications.",
    "Compare REST, gRPC, GraphQL on performance, schema evolution, ops complexity.",
    "Generate a JSON schema for a user profile with name, email, address, preferences.",
    "Refactor for clarity: def f(x, y, z): return x if y else (z if x else y)",
    "SQL query joining users, orders, products, top 5 customers by total order value.",
    "Describe trade-offs of microservices vs monolith in 200 words.",
    "List 5 GoF design patterns with a Python example of each.",
    "Explain CAP theorem with respect to distributed databases.",
    "Write a unit test for a Fibonacci function in Python.",
    "Generate a Dockerfile for a Python FastAPI app with PostgreSQL.",
    "Compare React Server Components vs traditional SSR.",
    "Explain backpropagation in neural networks in 200 words.",
    "Write a regex to match valid US phone numbers.",
    "Outline a CI/CD pipeline for a Rust project on Kubernetes.",
    "Describe OAuth 2.0 PKCE flow step by step.",
    "Compare PostgreSQL and SQLite for embedded application use.",
    "Walk through TCP three-way handshake with sequence numbers.",
    "Explain rate limiting algorithms: token bucket vs leaky bucket.",
    "Describe consistent hashing and where it is used.",
]

DENSE_TEXT_PROMPTS = [
    "Mixed scripts test for tokenizer compression: emoji symbols and special characters increase byte-to-token ratio meaningfully.",
    "Repeated punctuation patterns: !!!! ???? ---- ==== ____ #### **** test BPE handling of dense symbol runs.",
    "Mathematical-style notation: f(x) = sum from i=1 to n of (a_i * x_i)^2 + epsilon * sigma squared. Vocabulary stress test.",
    "Unicode-dense card suits and enclosed numbers test the tokenizer vocabulary coverage on rare codepoints.",
    "Nested JSON like {'a': {'b': {'c': {'d': 'deep'}}}} four levels increases tokenization due to syntax overhead.",
    "Base64 style dense alphanumeric: aGVsbG8gd29ybGQgdGhpcyBpcyBhIGxvbmcgYmFzZTY0LWxpa2Ugc3RyaW5n with no whitespace breaks.",
    "UUIDs like 550e8400-e29b-41d4-a716-446655440000 and 7c9e6679-7425-40de-944b-e07fc1f90ae7 stress BPE differently from prose.",
    "Hex-dense: 0xDEADBEEF 0xCAFEBABE 0x12345678 0xFEEDFACE 0xABCDEF01 0x0F0F0F0F mixed hex literal stress test.",
    "Long URL with many query params: example.com/api/v2/search?q=test&filter=type:document&sort=date_desc&page=1&limit=50",
    "Mixed deeply-nested JSON: " + json.dumps({"l1": {"l2": {"l3": {"l4": {"l5": "leaf", "data": [1,2,3,4,5,6,7,8,9,10]}}}}}),
    ]

ALL_PROMPTS = (
        [("simple_qa", p) for p in SIMPLE_QA]
        + [("tool_call", p) for p in TOOL_CALL_PROMPTS]
        + [("dense_text", p) for p in DENSE_TEXT_PROMPTS]
)

class StaticAnthropicEstimator:
    name = "static_2.0x"
    def estimate(self, prompt: str) -> int:
        return int(len(prompt) * STATIC_MARGIN)


class AdaptiveEstimator:
    name = "adaptive_eps_0.10"
    def __init__(self, epsilon: float = ADAPTIVE_EPSILON):
        assert 0.0 <= epsilon < 1.0
        self.epsilon = epsilon
        self.observed_max: float = 1.0
        self.call_count = 0

    def estimate(self, prompt: str) -> int:
        return int(len(prompt) * (self.observed_max + self.epsilon)) + 1  # ceil

    def record(self, prompt_chars: int, actual_input_tokens: int):
        if prompt_chars == 0:
            return
        ratio = actual_input_tokens / prompt_chars
        if ratio > self.observed_max:
            self.observed_max = ratio
        self.call_count += 1

def call_with_retry(client, prompt):
    for attempt, backoff in enumerate([0] + OVERLOAD_BACKOFF_S):
        if backoff > 0:
            time.sleep(backoff)
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < len(OVERLOAD_BACKOFF_S):
                continue
            raise

    raise RuntimeError("Exhausted retries")


def reserve_uc(prompt: str, estimator) -> Tuple[int, int]:
    est_input_tok = estimator.estimate(prompt)
    reserved = est_input_tok * INPUT_RATE_UC_PER_TOK + MAX_OUTPUT_TOKENS * OUTPUT_RATE_UC_PER_TOK

    return est_input_tok, reserved

def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: export ANTHROPIC_API_KEY=...", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "adaptive_vs_static_results.csv"
    summary_path = OUTPUT_DIR / "adaptive_vs_static_summary.csv"
    client = anthropic.Anthropic()

    static_est = StaticAnthropicEstimator()
    adaptive_est = AdaptiveEstimator(epsilon=ADAPTIVE_EPSILON)

    all_rows = []

    with open(results_path, "w", newline="") as f:
        fieldnames = [
            "estimator", "prompt_id", "category", "prompt_chars",
            "input_tokens", "output_tokens", "billed_uc",
            "reserved_uc", "effective_margin", "observed_max_after",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for prompt_id, (category, prompt) in enumerate(ALL_PROMPTS):
            prompt_chars = len(prompt)
            print(f"[{prompt_id+1:2d}/{len(ALL_PROMPTS)}] {category}: ", end="", flush=True)

            try:
                resp = call_with_retry(client, prompt)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            actual_input_tokens = resp.usage.input_tokens
            actual_output_tokens = resp.usage.output_tokens
            billed_uc = (
                    actual_input_tokens * INPUT_RATE_UC_PER_TOK
                    + actual_output_tokens * OUTPUT_RATE_UC_PER_TOK
            )

            _, static_reserved = reserve_uc(prompt, static_est)
            static_eff_margin = static_reserved / billed_uc if billed_uc > 0 else 0
            static_row = {
                "estimator": static_est.name,
                "prompt_id": prompt_id,
                "category": category,
                "prompt_chars": prompt_chars,
                "input_tokens": actual_input_tokens,
                "output_tokens": actual_output_tokens,
                "billed_uc": billed_uc,
                "reserved_uc": static_reserved,
                "effective_margin": round(static_eff_margin, 4),
                "observed_max_after": "",
            }
            writer.writerow(static_row)
            all_rows.append(static_row)

            _, adaptive_reserved = reserve_uc(prompt, adaptive_est)
            adaptive_est.record(prompt_chars, actual_input_tokens)
            adaptive_eff_margin = adaptive_reserved / billed_uc if billed_uc > 0 else 0
            adaptive_row = {
                "estimator": adaptive_est.name,
                "prompt_id": prompt_id,
                "category": category,
                "prompt_chars": prompt_chars,
                "input_tokens": actual_input_tokens,
                "output_tokens": actual_output_tokens,
                "billed_uc": billed_uc,
                "reserved_uc": adaptive_reserved,
                "effective_margin": round(adaptive_eff_margin, 4),
                "observed_max_after": round(adaptive_est.observed_max, 4),
            }
            writer.writerow(adaptive_row)
            all_rows.append(adaptive_row)
            f.flush()

            print(
                f"static_margin={static_eff_margin:.2f}x  "
                f"adaptive_margin={adaptive_eff_margin:.2f}x  "
                f"obs_max={adaptive_est.observed_max:.3f}"
            )

    by_est: Dict[str, List[dict]] = {}

    for r in all_rows:
        by_est.setdefault(r["estimator"], []).append(r)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "estimator", "n_calls", "median_effective_margin",
                "mean_effective_margin", "min_effective_margin",
                "max_effective_margin", "a1_violations",
                "median_capital_efficiency",
            ],
        )

        writer.writeheader()

        for est_name, rows in by_est.items():
            margins = [r["effective_margin"] for r in rows if r["billed_uc"] > 0]
            violations = sum(1 for r in rows if r["billed_uc"] > r["reserved_uc"])
            row = {
                "estimator": est_name,
                "n_calls": len(rows),
                "median_effective_margin": round(statistics.median(margins), 4),
                "mean_effective_margin": round(statistics.mean(margins), 4),
                "min_effective_margin": round(min(margins), 4),
                "max_effective_margin": round(max(margins), 4),
                "a1_violations": violations,
                "median_capital_efficiency": round(1.0 / statistics.median(margins), 4),
            }

            writer.writerow(row)
            print()
            print(f" {est_name} ")
            for k, v in row.items():
                if k != "estimator":
                    print(f"    {k}: {v}")

    print()
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        "Interpretation: lower effective_margin = tighter reservation "
        "= better capital efficiency, conditional on a1_violations == 0."
    )

if __name__ == "__main__":
    main()