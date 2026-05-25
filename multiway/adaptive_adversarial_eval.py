from __future__ import annotations
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import base64
import secrets

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: 'anthropic' package not installed. Run:")
    print("    pip install anthropic")
    sys.exit(1)


SONNET_MODEL = "claude-sonnet-4-5-20250929"
INPUT_RATE_UC_PER_TOK  = 3
OUTPUT_RATE_UC_PER_TOK = 15
MAX_OUTPUT_TOKENS = 100
OUTPUT_RESERVATION_UC = MAX_OUTPUT_TOKENS * OUTPUT_RATE_UC_PER_TOK

STATIC_MARGIN = 2.0
ADAPTIVE_EPSILON = 0.10
OUT_DIR = Path(__file__).resolve().parent / "sweep_results"

def _nested_tool_schemas(n: int) -> List[dict]:
    out = []
    for i in range(n):
        depth = 1
        n_keys = 20 + (i * 3) % 60   # 20..79 keys
        schema = _build_wide_schema(depth, n_keys, salt=i)
        prompt = (
                "You are an agent. Here is your tool schema (do not call the tool, "
                "just acknowledge it):\n\n"
                "```json\n"
                + json.dumps(schema, indent=2)
                + "\n```\n\n"
                  "Acknowledge in one sentence."
        )
        out.append({"category": "nested_tool_schemas", "prompt_id": f"nts_{i:02d}", "text": prompt})
    return out

def _build_wide_schema(depth: int, n_keys: int, salt: int) -> dict:
    if depth <= 0:
        return {"type": "string", "salt": salt}
    return {
        "type": "object",
        "properties": {
            f"k{j}_{salt}": _build_wide_schema(depth - 1, n_keys, salt + j * 7)
            for j in range(n_keys)
        },
        "required": [f"k{j}_{salt}" for j in range(n_keys)],
    }

def _unicode_combining(n: int) -> List[dict]:
    devanagari_base = "आपको हमारे साथ इस यात्रा में शामिल होकर बहुत खुशी हो रही है। "
    arabic_base     = "نحن سعداء جدا بانضمامك إلينا في هذه الرحلة العلمية المثيرة. "
    hebrew_base     = "אנו שמחים מאוד שהצטרפת אלינו במסע המחקרי המרתק הזה. "
    cjk_base        = "我们非常高兴你能加入我们这次精彩的研究之旅。"
    bases = [devanagari_base, arabic_base, hebrew_base, cjk_base]
    out = []
    for i in range(n):
        base = bases[i % len(bases)]
        text = base * (8 + (i % 5))
        prompt = (
                "Translate the following text to English. Reply with the translation only.\n\n"
                + text
        )
        out.append({"category": "unicode_combining_marks",
                    "prompt_id": f"ucm_{i:02d}", "text": prompt})
    return out

def _base64_dense(n: int) -> List[dict]:
    out = []
    for i in range(n):
        size = 500 + (i % 5) * 200   # 500..1300 bytes
        raw = secrets.token_bytes(size)
        b64 = base64.b64encode(raw).decode()
        prompt = (
                "I have a base64-encoded payload. Tell me roughly how many bytes it "
                "would decode to (just give a number, don't decode it):\n\n" + b64
        )
        out.append({"category": "base64_dense", "prompt_id": f"b64_{i:02d}", "text": prompt})
    return out

def build_corpus() -> List[dict]:
    prompts = _nested_tool_schemas(20) + _unicode_combining(15) + _base64_dense(15)
    print(f"Built adversarial corpus: {len(prompts)} prompts across "
          f"{len(set(p['category'] for p in prompts))} categories")
    return prompts

@dataclass
class EstimatorReport:
    estimator: str
    prompt_id: str
    category: str
    prompt_chars: int
    input_tokens: int
    output_tokens: int
    billed_uc: int
    reserved_uc: int
    effective_margin: float
    observed_max_after: float

class StaticEstimator:
    def __init__(self, margin: float = STATIC_MARGIN):
        self.margin = margin
    def reserve_for(self, prompt_chars: int) -> int:
        return int(prompt_chars * self.margin * INPUT_RATE_UC_PER_TOK) + OUTPUT_RESERVATION_UC
    def record(self, prompt_chars: int, actual_input_tokens: int):
        pass
    @property
    def observed_max(self) -> float:
        return self.margin  # report the static margin

class AdaptiveEstimator:
    def __init__(self, epsilon: float = ADAPTIVE_EPSILON):
        self.epsilon = epsilon
        self._observed_max = 1.0
        self.call_count = 0
    def reserve_for(self, prompt_chars: int) -> int:
        eff_ratio = self._observed_max + self.epsilon
        return int(prompt_chars * eff_ratio * INPUT_RATE_UC_PER_TOK) + OUTPUT_RESERVATION_UC
    def record(self, prompt_chars: int, actual_input_tokens: int):
        if prompt_chars > 0:
            ratio = actual_input_tokens / prompt_chars
            if ratio > self._observed_max:
                self._observed_max = ratio
        self.call_count += 1
    @property
    def observed_max(self) -> float:
        return self._observed_max

def run_one(estimator_label: str, estimator, prompts: List[dict],
            client: Anthropic) -> List[EstimatorReport]:
    reports = []
    for p in prompts:
        prompt = p["text"]
        chars = len(prompt.encode("utf-8"))
        reserved_uc = estimator.reserve_for(chars)
        # API call
        try:
            resp = client.messages.create(
                model=SONNET_MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"  [ERR] {p['prompt_id']}: {e}")
            continue

        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        billed_uc = in_tok * INPUT_RATE_UC_PER_TOK + out_tok * OUTPUT_RATE_UC_PER_TOK
        eff_margin = (reserved_uc / billed_uc) if billed_uc > 0 else float("inf")

        estimator.record(chars, in_tok)

        reports.append(EstimatorReport(
            estimator=estimator_label,
            prompt_id=p["prompt_id"], category=p["category"],
            prompt_chars=chars,
            input_tokens=in_tok, output_tokens=out_tok,
            billed_uc=billed_uc, reserved_uc=reserved_uc,
            effective_margin=round(eff_margin, 4),
            observed_max_after=round(estimator.observed_max, 4),
        ))
        print(f"  [{estimator_label}] {p['prompt_id']:>10s} "
              f"({p['category']:>22s}): in_tok={in_tok:>4d} chars={chars:>5d} "
              f"ratio={in_tok/max(chars,1):.3f} reserved={reserved_uc} "
              f"billed={billed_uc} margin={eff_margin:.2f}x "
              f"obs_max={estimator.observed_max:.3f}")
        time.sleep(0.2)
    return reports


def summarise(reports: List[EstimatorReport]) -> dict:
    by_est = {}
    for r in reports:
        by_est.setdefault(r.estimator, []).append(r)
    summaries = []
    for est, rs in by_est.items():
        margins = [r.effective_margin for r in rs if r.effective_margin != float("inf")]
        a1_viol = sum(1 for r in rs if r.billed_uc > r.reserved_uc)
        cap_eff = [1.0 / m for m in margins if m > 0]
        summaries.append({
            "estimator": est,
            "n_calls": len(rs),
            "median_effective_margin": round(statistics.median(margins), 4) if margins else None,
            "mean_effective_margin":   round(statistics.mean(margins), 4)   if margins else None,
            "min_effective_margin":    round(min(margins), 4)               if margins else None,
            "max_effective_margin":    round(max(margins), 4)               if margins else None,
            "a1_violations": a1_viol,
            "median_capital_efficiency": round(statistics.median(cap_eff), 4) if cap_eff else None,
            "final_observed_max": round(rs[-1].observed_max_after, 4) if rs else None,
        })
    return summaries


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Source your shell rc file:")
        print("    source ~/.zshrc   # or .bashrc")
        sys.exit(1)
    client = Anthropic()

    prompts = build_corpus()

    print()
    print("=== STATIC estimator pass ===")
    static_est = StaticEstimator(margin=STATIC_MARGIN)
    static_reports = run_one("static_2.0x", static_est, prompts, client)

    print()
    print("=== ADAPTIVE estimator pass (eps=0.10) ===")
    adaptive_est = AdaptiveEstimator(epsilon=ADAPTIVE_EPSILON)
    adaptive_reports = run_one("adaptive_eps_0.10", adaptive_est, prompts, client)

    all_reports = static_reports + adaptive_reports
    print()
    print(f"Total calls: {len(all_reports)} (expected {2*len(prompts)})")

    results_path = OUT_DIR / "adaptive_adversarial_results.csv"
    with open(results_path, "w", newline="") as f:
        if all_reports:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_reports[0]).keys()))
            w.writeheader()
            for r in all_reports:
                w.writerow(asdict(r))
    print(f"  -> {results_path} ({len(all_reports)} rows)")

    summary = summarise(all_reports)
    summary_path = OUT_DIR / "adaptive_adversarial_summary.csv"
    with open(summary_path, "w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            for s in summary:
                w.writerow(s)
    print(f"  -> {summary_path} ({len(summary)} rows)")

    print()
    print("=== SUMMARY ===")
    for s in summary:
        print(f"  {s['estimator']}:")
        print(f"    n_calls = {s['n_calls']}")
        print(f"    median margin = {s['median_effective_margin']}x")
        print(f"    mean margin   = {s['mean_effective_margin']}x")
        print(f"    A1 violations = {s['a1_violations']} / {s['n_calls']}")
        print(f"    final observed_max = {s['final_observed_max']}")

    adaptive_summary = next((s for s in summary if s["estimator"].startswith("adaptive")), None)
    if adaptive_summary and adaptive_summary["final_observed_max"] is not None:
        if adaptive_summary["final_observed_max"] > 1.05:
            print()
            print(f"  GOOD: Adaptive estimator's observed_max climbed to "
                  f"{adaptive_summary['final_observed_max']} (above the 1.0 floor). "
                  "The learning path WAS exercised on this corpus.")
        else:
            print()
            print(f"  CAVEAT: Adaptive estimator's observed_max stayed at "
                  f"{adaptive_summary['final_observed_max']}. The learning path was "
                  "still not exercised. Consider stronger adversarial prompts.")

if __name__ == "__main__":
    main()