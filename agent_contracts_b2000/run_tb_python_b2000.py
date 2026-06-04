from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from collections import Counter

try:
    import litellm
    _HAVE_LITELLM = True
except ImportError:
    _HAVE_LITELLM = False

from run_ac_b2000 import (
    MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS, MAX_AGENT_STEPS,
    CAP_USD, UC_PER_USD, PRICE_IN_PER_MTOK_USD, PRICE_OUT_PER_MTOK_USD,
    MARGIN, DEFAULT_PROMPTS, estimate_call_cost_usd,
    estimate_input_byte_length, TrialResult,
)

def run_one_trial(trial_id: int, prompts: dict, dry_run: bool = False, verbose: bool = False) -> TrialResult:
    t0 = time.time()
    remaining_usd = CAP_USD
    spend_usd = 0.0
    system = prompts["system"]
    tools_def = prompts["tools"]
    tool_error = prompts["tool_error"]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts["user"]},
    ]
    
    openai_tools = [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    } for t in tools_def]

    steps_admitted = 0
    pre_flight_refusals = 0
    refusal_reservation_uc = 0

    for step in range(MAX_AGENT_STEPS):
        estimate_usd = estimate_call_cost_usd(messages, tools_def, system)

        if estimate_usd > remaining_usd:
            pre_flight_refusals = 1
            refusal_reservation_uc = int(round(estimate_usd * UC_PER_USD))
            outcome = "admit_then_refuse" if steps_admitted > 0 else "refuse_pre_flight"
            break

        if dry_run:
            input_tokens = max(estimate_input_byte_length(messages, tools_def, system) // 4, 50)
            output_tokens = 80
            is_tool_use = step < 3
            tool_call_id = f"call_dry_{step}"
            tool_args_json = json.dumps({"query": "SELECT COUNT(*) FROM users"})
        else:
            if not _HAVE_LITELLM:
                return TrialResult(trial_id, "import_error", 0, 0,
                                   int(CAP_USD * UC_PER_USD), int(CAP_USD * UC_PER_USD),
                                   0, 0, time.time() - t0, "litellm not installed")
            try:
                resp = litellm.completion(
                    model=f"anthropic/{MODEL}",
                    messages=messages,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=TEMPERATURE,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                input_tokens = resp.usage.prompt_tokens
                output_tokens = resp.usage.completion_tokens
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                is_tool_use = len(tool_calls) > 0

                if is_tool_use:
                    tc = tool_calls[0]
                    tool_call_id = tc.id
                    tool_args_json = tc.function.arguments
                else:
                    tool_call_id = None
                    tool_args_json = None
                if verbose:
                    print(f"      step {step}: in={input_tokens} out={output_tokens} "
                          f"is_tool={is_tool_use}")
            except Exception as e:
                if verbose:
                    print(f"      step {step}: EXCEPTION {type(e).__name__}: {e}")
                err_str = f"{type(e).__name__}: {e}"[:300]
                return TrialResult(trial_id, "api_error", steps_admitted,
                                   int(round(spend_usd * UC_PER_USD)),
                                   int(CAP_USD * UC_PER_USD),
                                   int(round(remaining_usd * UC_PER_USD)),
                                   pre_flight_refusals, refusal_reservation_uc,
                                   time.time() - t0, err_str)

        actual_usd = (input_tokens * PRICE_IN_PER_MTOK_USD / 1_000_000
                      + output_tokens * PRICE_OUT_PER_MTOK_USD / 1_000_000)
        spend_usd += actual_usd
        remaining_usd = CAP_USD - spend_usd
        steps_admitted += 1

        if not is_tool_use:
            outcome = "model_terminated"
            break

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "arguments": tool_args_json or "{}",
                },
            }],
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_error,
        })

        if spend_usd > CAP_USD:
            outcome = "violation"
            break
    else:
        outcome = "step_limit"

    return TrialResult(
        trial_id=trial_id,
        outcome=outcome,
        steps_admitted=steps_admitted,
        cumulative_spend_uc=int(round(spend_usd * UC_PER_USD)),
        cap_uc=int(CAP_USD * UC_PER_USD),
        final_remaining_uc=int(round(remaining_usd * UC_PER_USD)),
        pre_flight_refusals=pre_flight_refusals,
        refusal_reservation_uc=refusal_reservation_uc,
        elapsed_s=time.time() - t0,
    )

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--output", type=str, default="tb_python_b2000_results.csv")
    p.add_argument("--prompts", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    prompts = DEFAULT_PROMPTS

    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Use --dry-run for offline test.", file=sys.stderr)
        sys.exit(1)

    init_messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]},
    ]
    
    init_bytes = estimate_input_byte_length(init_messages, prompts["tools"],
                                            prompts["system"])
    init_estimate_uc = int(round(estimate_call_cost_usd(init_messages, prompts["tools"], prompts["system"]) * UC_PER_USD))
    cap_uc = int(CAP_USD * UC_PER_USD)
    print(f"TB Python head-to-head, B_0 = {cap_uc} uc (${CAP_USD:.4f})")
    print(f"  Model: {MODEL}, T={TEMPERATURE}, max_output={MAX_OUTPUT_TOKENS}")
    print(f"  Discipline: plain integer counter + pre-flight check")
    print(f"  Initial payload: {init_bytes} bytes, "
          f"step 0 estimate: {init_estimate_uc} uc")
    print(f"  Trials: {args.trials}, dry-run: {args.dry_run}")
    print()

    results = []

    for i in range(args.trials):
        r = run_one_trial(i, prompts, dry_run=args.dry_run, verbose=args.verbose)
        results.append(r)

        if not args.quiet:
            err_tag = f" err={r.error[:60]!r}" if r.error else ""
            print(f"  trial {i:3d}: {r.outcome:22s} "
                  f"steps={r.steps_admitted} "
                  f"spend={r.cumulative_spend_uc}/{r.cap_uc}uc "
                  f"refusals={r.pre_flight_refusals} "
                  f"({r.elapsed_s:.2f}s){err_tag}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].__dict__.keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)
    print(f"\nWrote {len(results)} rows to {out.resolve()}")

    outcomes = Counter(r.outcome for r in results)
    print("\n Summary ")

    for k, v in outcomes.most_common():
        print(f"  {k}: {v}/{len(results)}")
    overshoots = sum(1 for r in results if r.cumulative_spend_uc > r.cap_uc)
    print(f"  overshoots (spend > cap): {overshoots}/{len(results)}")

if __name__ == "__main__":
    main()