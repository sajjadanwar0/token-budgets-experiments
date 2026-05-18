#!/usr/bin/env python3
"""Adversarial AnthropicEstimator audit harness (CORRECTED v2).

Place at: experiments/anthropic_adversarial/run.py in
token-budgets-experiments.

CRITICAL difference from v1 (which was wrong):
    v1 tested `count_tokens(prompt) >= input_tokens(prompt)`. Since
    both numbers come from the same Anthropic tokenizer, the ratio
    was always exactly 1.0000 — a tautology, not an audit.

    v2 tests `byte_length(serialized_prompt) * safety_margin >=
    input_tokens(prompt)`, which is what the AnthropicEstimator
    actually does in src/estimator.rs (ByteLength base x 1.05
    margin by default). This produces meaningful ratios because
    byte-length and tokenizer-output are independent measurements
    of the same content. The audit can therefore detect either:
      (a) prompt classes where byte-length under-counts vs billing
          (A1 violation under the configured margin), or
      (b) the empirical headroom the default margin actually
          provides on adversarial inputs.

Byte-length computation matches budget-spike's
estimate_input_bytes() exactly:
  - sum of message.role and message.content lengths
  - sum of tool name, description, and schema_json lengths
  - 64-byte envelope slack per message
  - applied to all classes uniformly

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run.py [--runs 5] [--margin 1.05]

Output: results.csv with per-run details (class, byte_length,
        estimate_with_margin, billed_input, ratio, a1_holds)
        results.md with summary table for paste into the paper

Cost: approximately $0.50 for default 35 runs.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic>=0.39")
    sys.exit(1)


MODEL = "claude-haiku-4-5-20251001"
ENVELOPE_BYTES_PER_MESSAGE = 64  # matches tc_live_harness's slack constant


# -----------------------------------------------------------------------
# Adversarial prompt classes (unchanged from v1)
# -----------------------------------------------------------------------

def large_tool_definition(n_tools: int = 1, desc_chars: int = 4000) -> Dict[str, Any]:
    return {
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [
            {
                "name": f"tool_{i}",
                "description": "A" * desc_chars,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        f"arg_{j}": {"type": "string", "description": "B" * 200}
                        for j in range(20)
                    },
                    "required": [f"arg_{j}" for j in range(10)],
                },
            }
            for i in range(n_tools)
        ],
    }


def long_system_prompt(chars: int = 8000) -> Dict[str, Any]:
    return {
        "system": "You are a helpful assistant. " + ("X " * (chars // 2)),
        "messages": [{"role": "user", "content": "Hi"}],
    }


def multi_turn_history(turns: int = 15, per_turn_chars: int = 500) -> Dict[str, Any]:
    msgs = []
    for i in range(turns):
        msgs.append({"role": "user", "content": ("U " * (per_turn_chars // 2)) + f"turn {i}"})
        msgs.append({"role": "assistant", "content": ("A " * (per_turn_chars // 2)) + f"resp {i}"})
    msgs.append({"role": "user", "content": "Final question?"})
    return {"messages": msgs}


def multi_tool_results(n_results: int = 10, per_result_chars: int = 300) -> Dict[str, Any]:
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": "Compute something."}]
    for i in range(n_results):
        msgs.append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"tu_{i}", "name": "tool_a", "input": {"x": i}}
            ],
        })
        msgs.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"tu_{i}", "content": "R" * per_result_chars}
            ],
        })
    msgs.append({"role": "user", "content": "Summarize."})
    return {
        "messages": msgs,
        "tools": [{
            "name": "tool_a",
            "description": "Does stuff.",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        }],
    }


def cache_control_breakpoints() -> Dict[str, Any]:
    return {
        "system": [
            {"type": "text",
             "text": "You are a helpful assistant. " + ("Z " * 200),
             "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "Hi"}],
    }


def nested_tool_schema(depth: int = 5) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "string"}
    for _ in range(depth):
        schema = {"type": "object", "properties": {"nested": schema}}
    return {
        "messages": [{"role": "user", "content": "Use the tool."}],
        "tools": [{"name": "nested", "description": "nested", "input_schema": schema}],
    }


def unicode_dense_tool_desc() -> Dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "Process data."}],
        "tools": [{
            "name": "unicode_tool",
            "description": "处理数据 🚀🎉🔥 用户请求 obtenir résultats " * 200,
            "input_schema": {"type": "object", "properties": {}},
        }],
    }


ADVERSARIAL_CLASSES = {
    "large_tool_def":          large_tool_definition,
    "long_system_prompt":      long_system_prompt,
    "multi_turn_history":      multi_turn_history,
    "multi_tool_results":      multi_tool_results,
    "cache_control":           cache_control_breakpoints,
    "nested_tool_schema":      nested_tool_schema,
    "unicode_dense_tool_desc": unicode_dense_tool_desc,
}


# -----------------------------------------------------------------------
# Byte-length estimator (mirrors tc_live_harness's estimate_input_bytes)
# -----------------------------------------------------------------------

def _stringify_content(content: Any) -> str:
    """Anthropic message content can be a string or a list of typed blocks.
    Return a UTF-8 string for byte-counting that captures all text. Tool
    blocks are stringified by their JSON content so they contribute to
    byte-length. This matches the conservative spirit of byte-length:
    everything serializable is counted."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # text blocks, tool_use blocks, tool_result blocks, etc
                for k, v in block.items():
                    if isinstance(v, str):
                        parts.append(v)
                    else:
                        parts.append(json.dumps(v, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def byte_length_of_prompt(prompt: Dict[str, Any]) -> int:
    """Byte-length of the prompt as the AnthropicEstimator's base would
    compute it. Mirrors tc_live_harness's estimate_input_bytes:
      - sum of role/content UTF-8 byte lengths
      - sum of tool name/description/schema_json byte lengths
      - 64-byte envelope slack per message
    The schema is serialized to compact JSON (no whitespace) to match
    the Rust side's serde_json::Value::to_string default.
    """
    total = 0

    # System
    system = prompt.get("system", "")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                for k, v in block.items():
                    if isinstance(v, str):
                        total += len(v.encode("utf-8"))
                    else:
                        total += len(json.dumps(v, ensure_ascii=False).encode("utf-8"))
    else:
        total += len(system.encode("utf-8"))

    # Messages
    messages = prompt.get("messages", [])
    for m in messages:
        role = m.get("role", "")
        content_str = _stringify_content(m.get("content", ""))
        total += len(role.encode("utf-8"))
        total += len(content_str.encode("utf-8"))
    total += len(messages) * ENVELOPE_BYTES_PER_MESSAGE

    # Tools
    for t in prompt.get("tools", []):
        total += len(t.get("name", "").encode("utf-8"))
        total += len(t.get("description", "").encode("utf-8"))
        schema_json = json.dumps(t.get("input_schema", {}), ensure_ascii=False,
                                 separators=(",", ":"))
        total += len(schema_json.encode("utf-8"))

    return total


def anthropic_estimator_output(prompt: Dict[str, Any], margin: float) -> int:
    """Match AnthropicEstimator::estimate() in src/estimator.rs:
        raw = ByteLength.estimate(prompt)
        return ceil(raw * margin)
    """
    raw = byte_length_of_prompt(prompt)
    import math
    return int(math.ceil(raw * margin))


# -----------------------------------------------------------------------
# Main audit loop
# -----------------------------------------------------------------------

def run_audit(runs_per_class: int, margin: float, sleep_secs: float = 0.5) -> List[Dict]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results: List[Dict] = []

    print(f"AnthropicEstimator adversarial audit "
          f"(byte-length x {margin} margin)")
    print(f"  model:    {MODEL}")
    print(f"  classes:  {len(ADVERSARIAL_CLASSES)}")
    print(f"  runs/cls: {runs_per_class}")
    print()

    for class_name, builder in ADVERSARIAL_CLASSES.items():
        print(f"=== {class_name} ===")
        for run_i in range(runs_per_class):
            prompt = builder()

            # AnthropicEstimator output (= byte_length x margin, ceil)
            bl = byte_length_of_prompt(prompt)
            estimate = anthropic_estimator_output(prompt, margin)

            # Provider-billed input via max_tokens=1 call
            kwargs: Dict[str, Any] = {
                "model": MODEL,
                "max_tokens": 1,
                "messages": prompt["messages"],
            }
            if "system" in prompt:
                kwargs["system"] = prompt["system"]
            if "tools" in prompt:
                kwargs["tools"] = prompt["tools"]

            try:
                resp = client.messages.create(**kwargs)
                billed_input  = resp.usage.input_tokens
                cache_create  = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
                cache_read    = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
                total_billed  = billed_input + cache_create
            except Exception as e:
                print(f"  Run {run_i}: API ERROR {type(e).__name__}: {e}")
                continue

            ratio = estimate / max(total_billed, 1)
            a1_holds = ratio >= 1.0
            margin_needed = (total_billed / max(bl, 1)) if bl > 0 else float("inf")

            results.append({
                "class": class_name,
                "run": run_i,
                "byte_length": bl,
                "estimate_with_margin": estimate,
                "billed_input": billed_input,
                "cache_creation": cache_create,
                "cache_read": cache_read,
                "total_billed": total_billed,
                "ratio": round(ratio, 4),
                "a1_holds": a1_holds,
                "min_margin_for_a1": round(margin_needed, 4),
            })

            marker = "OK" if a1_holds else "A1 VIOLATION"
            print(f"  Run {run_i}: bl={bl}, est(x{margin})={estimate}, "
                  f"billed={total_billed}, ratio={ratio:.4f} [{marker}]")
            time.sleep(sleep_secs)

    return results


def write_results(results: List[Dict], margin: float) -> None:
    if not results:
        print("No results collected.")
        return

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    by_class: Dict[str, List[Dict]] = {}
    for r in results:
        by_class.setdefault(r["class"], []).append(r)

    with open("results.md", "w") as f:
        f.write("# Adversarial AnthropicEstimator Audit (v2)\n\n")
        f.write(f"Tested: AnthropicEstimator with ByteLength base x "
                f"{margin} safety_margin\n")
        f.write(f"Provider: Anthropic claude-haiku-4-5-20251001 "
                f"(max_tokens=1, prompt_tokens recovered from usage block)\n\n")

        f.write(f"Total runs: {len(results)}\n")
        violations = [r for r in results if not r["a1_holds"]]
        f.write(f"A1 holds: {len(results) - len(violations)}/{len(results)}\n")
        f.write(f"A1 violations: {len(violations)}\n\n")

        f.write("## Per-class summary\n\n")
        f.write("| Class | N | Min ratio | Mean ratio | Max ratio | Min margin needed | A1 holds |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for class_name, rows in by_class.items():
            ratios = [r["ratio"] for r in rows]
            margins_needed = [r["min_margin_for_a1"] for r in rows]
            holds = sum(1 for r in rows if r["a1_holds"])
            f.write(f"| {class_name} | {len(rows)} | "
                    f"{min(ratios):.4f} | "
                    f"{sum(ratios)/len(ratios):.4f} | "
                    f"{max(ratios):.4f} | "
                    f"{max(margins_needed):.4f} | "
                    f"{holds}/{len(rows)} |\n")

        all_min_margins = [r["min_margin_for_a1"] for r in results]
        worst_margin = max(all_min_margins)
        f.write(f"\n## Margin headroom analysis\n\n")
        f.write(f"Configured margin: {margin}x\n\n")
        f.write(f"Worst-case min margin needed across all "
                f"{len(results)} runs: **{worst_margin:.4f}x**\n\n")
        if worst_margin <= margin:
            headroom = (margin - worst_margin) * 100
            f.write(f"The configured {margin}x margin is adequate "
                    f"with **{headroom:.1f} percentage points of headroom** "
                    f"on the worst-case prompt class.\n\n")
            f.write(f"Recommended `AnthropicEstimator::safety_margin`: "
                    f"**{margin}** (current default) is sufficient. "
                    f"A tighter bound of **{max(1.02, worst_margin + 0.02):.4f}** "
                    f"would still preserve A1 with 2% headroom on the "
                    f"audited prompt classes.\n")
        else:
            shortfall = (worst_margin - margin) * 100
            f.write(f"**The configured {margin}x margin is INSUFFICIENT** "
                    f"by {shortfall:.1f} percentage points on the worst-case "
                    f"prompt class.\n\n")
            f.write(f"Recommended `AnthropicEstimator::safety_margin`: "
                    f"**{worst_margin + 0.02:.4f}** (worst-case + 2% headroom).\n\n")
            f.write("Worst-case classes:\n")
            for class_name, rows in by_class.items():
                cms = [r["min_margin_for_a1"] for r in rows]
                if max(cms) > margin:
                    f.write(f"  - {class_name}: min margin needed = "
                            f"{max(cms):.4f}x ({len(rows)} runs)\n")

    print("\n=== Wrote results.csv and results.md ===")
    print(f"Worst min-margin needed across all runs: {worst_margin:.4f}x")
    if worst_margin <= margin:
        print(f"Configured {margin}x margin: ADEQUATE "
              f"(headroom: {(margin - worst_margin)*100:.1f} pp)")
    else:
        print(f"Configured {margin}x margin: INSUFFICIENT "
              f"(shortfall: {(worst_margin - margin)*100:.1f} pp)")
        print(f"Recommend: safety_margin = {worst_margin + 0.02:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per class (default 5)")
    parser.add_argument("--margin", type=float, default=1.05,
                        help="AnthropicEstimator safety_margin to test "
                             "(default 1.05, matching src/estimator.rs default)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds between requests")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    results = run_audit(args.runs, args.margin, args.sleep)
    write_results(results, args.margin)


if __name__ == "__main__":
    main()