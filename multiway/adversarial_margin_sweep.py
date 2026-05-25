import argparse
import csv
import json
import math
import os
import sys
import time
from anthropic import Anthropic

ANTHROPIC_HAIKU_4_5 = "claude-haiku-4-5-20251001"
PRICING_UC_PER_TOKEN = {"input": 1, "output": 5}

def make_large_tool_def():
    """4,000-character description + 20 nested arguments."""
    desc = "A complex tool that performs an operation. " * 100
    desc = desc[:4000]
    args = {f"arg{i}": {"type": "string", "description": f"argument {i}"}
            for i in range(20)}
    tool = {
        "name": "complex_op",
        "description": desc,
        "input_schema": {"type": "object", "properties": args,
                         "required": list(args.keys())},
    }
    return {
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Call complex_op with valid arguments."}],
        "tools": [tool],
    }


def make_long_system_prompt():
    """8,000-character system content."""
    sys_text = ("You are a sophisticated AI assistant. " * 200)[:8000]
    return {
        "system": sys_text,
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": None,
    }


def make_multi_turn_history():
    """15 turns, 500 chars per turn."""
    user_msg = "Please summarise this conversation. " * 15
    user_msg = user_msg[:500]
    asst_msg = "I have summarised it for you. " * 17
    asst_msg = asst_msg[:500]
    messages = []
    for i in range(7):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": user_msg})
    return {
        "system": "You are a helpful summariser.",
        "messages": messages,
        "tools": None,
    }


def make_multi_tool_results():
    """10 sequential tool-call/result pairs, 300 chars per result."""
    result_text = "Result data: " + ("x" * 285)
    messages = [{"role": "user", "content": "Call get_data 10 times sequentially."}]
    for i in range(10):
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"tool_{i}",
                 "name": "get_data", "input": {"i": i}}
            ],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"tool_{i}",
                 "content": result_text}
            ],
        })
    tool = {
        "name": "get_data",
        "description": "Fetch data for index i.",
        "input_schema": {"type": "object",
                         "properties": {"i": {"type": "integer"}},
                         "required": ["i"]},
    }
    return {
        "system": "You are a data agent.",
        "messages": messages,
        "tools": [tool],
    }


def make_cache_control():
    """Anthropic's prompt-caching with ephemeral cache regions."""
    system_blocks = [
        {"type": "text",
         "text": ("Context document. " * 400)[:4000],
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Now answer the user's question."},
    ]
    return {
        "system": system_blocks,
        "messages": [{"role": "user", "content": "What is the document about?"}],
        "tools": None,
    }


def make_nested_tool_schema():
    """5-level recursive JSON schema."""
    schema = {"type": "object",
              "properties": {"leaf": {"type": "string"}}}
    for level in range(5):
        schema = {"type": "object",
                  "properties": {
                      f"level_{level}_a": schema,
                      f"level_{level}_b": schema,
                  }}
    tool = {
        "name": "process_nested",
        "description": "Process a deeply nested structure.",
        "input_schema": schema,
    }
    return {
        "system": "You are a structured agent.",
        "messages": [{"role": "user", "content": "Call process_nested with sample data."}],
        "tools": [tool],
    }


def make_unicode_dense_tool_desc():
    """CJK + emoji in tool metadata."""
    desc = "工具用于处理数据 📊📈📉 复杂的计算流程 🔢🔣 " * 50
    tool = {
        "name": "unicode_tool",
        "description": desc,
        "input_schema": {"type": "object",
                         "properties": {"x": {"type": "string"}}},
    }
    return {
        "system": "You are an assistant with Unicode-dense tools.",
        "messages": [{"role": "user", "content": "Call unicode_tool with appropriate input."}],
        "tools": [tool],
    }


ADVERSARIAL_CLASSES = {
    "large_tool_def": make_large_tool_def,
    "long_system_prompt": make_long_system_prompt,
    "multi_turn_history": make_multi_turn_history,
    "multi_tool_results": make_multi_tool_results,
    "cache_control": make_cache_control,
    "nested_tool_schema": make_nested_tool_schema,
    "unicode_dense_tool_desc": make_unicode_dense_tool_desc,
}


# ---------------------------------------------------------------------------
# A1 evaluation
# ---------------------------------------------------------------------------

def serialize_request_body(payload):
    """Same serialisation pattern as runner.py for byte_length."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def evaluate_a1(client, class_name, factory, margin, max_retries=5):
    """Return dict with predicted/actual/holds."""
    spec = factory()
    payload = {
        "model": ANTHROPIC_HAIKU_4_5,
        "max_tokens": 1,
        "system": spec["system"],
        "messages": spec["messages"],
    }
    if spec.get("tools"):
        payload["tools"] = spec["tools"]

    body = serialize_request_body(payload)
    byte_len = len(body.encode("utf-8"))
    predicted_input_uc = math.ceil(byte_len * margin)

    # count_tokens to get billed input
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": ANTHROPIC_HAIKU_4_5,
                "system": spec["system"],
                "messages": spec["messages"],
            }
            if spec.get("tools"):
                kwargs["tools"] = spec["tools"]
            resp = client.messages.count_tokens(**kwargs)
            actual_input_tokens = resp.input_tokens
            err = None
            break
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ("529", "Overloaded", "overloaded_error")):
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt + 1, 30))
                    continue
            err = str(e)
            actual_input_tokens = 0
            break

    actual_input_uc = actual_input_tokens * PRICING_UC_PER_TOKEN["input"]
    a1_holds = (predicted_input_uc >= actual_input_uc) and (err is None)
    ratio = predicted_input_uc / actual_input_uc if actual_input_uc > 0 else float("inf")

    return {
        "class": class_name,
        "margin": margin,
        "byte_length": byte_len,
        "predicted_input_uc": predicted_input_uc,
        "actual_input_tokens": actual_input_tokens,
        "actual_input_uc": actual_input_uc,
        "ratio": round(ratio, 4),
        "a1_holds": a1_holds,
        "error": err or "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margins", type=float, nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--n-trials", type=int, default=5,
        help="Trials per (margin, class) cell")
    parser.add_argument("--classes", nargs="+", default=list(ADVERSARIAL_CLASSES.keys()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    client = Anthropic()

    total = len(args.margins) * len(args.classes) * args.n_trials
    print(f"{'='*78}")
    print(f"Adversarial margin sweep")
    print(f"{'='*78}")
    print(f"  Margins:  {args.margins}")
    print(f"  Classes:  {args.classes}")
    print(f"  Trials/cell: {args.n_trials}")
    print(f"  Total trials: {total}")
    print(f"{'='*78}")

    rows = []
    cell = 0
    for class_name in args.classes:
        factory = ADVERSARIAL_CLASSES[class_name]
        for margin in args.margins:
            for trial in range(args.n_trials):
                cell += 1
                row = evaluate_a1(client, class_name, factory, margin)
                row["trial"] = trial
                rows.append(row)
                holds_str = "HOLDS" if row["a1_holds"] else "FAILS"
                print(f"  [{cell:>3}/{total:>3}] class={class_name[:22]:<22} "
                      f"margin={margin:.1f}x trial={trial} "
                      f"bytes={row['byte_length']:>6} "
                      f"pred={row['predicted_input_uc']:>6}uc "
                      f"actual={row['actual_input_uc']:>5}uc "
                      f"ratio={row['ratio']:>5} {holds_str}")
                if args.sleep > 0:
                    time.sleep(args.sleep)

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.output}")

    # ---- Summary: A1 pass rate per (margin, class) ----
    print(f"\n{'='*78}")
    print("SUMMARY: A1 pass rate (fraction of trials with predicted >= actual)")
    print(f"{'='*78}")
    print(f"{'Class':<24}", end="")
    for margin in args.margins:
        print(f"  {margin:>4}x", end="")
    print()
    print("-" * 78)
    for class_name in args.classes:
        print(f"{class_name[:24]:<24}", end="")
        for margin in args.margins:
            cell_rows = [r for r in rows
                         if r["class"] == class_name and r["margin"] == margin]
            holds = sum(1 for r in cell_rows if r["a1_holds"])
            n = len(cell_rows)
            print(f"  {holds:>2}/{n:<2}", end="")
        print()


if __name__ == "__main__":
    main()