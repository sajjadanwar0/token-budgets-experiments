import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "llama3.2:latest"
N_RUNS = 10
RECURSION_LIMIT = 20
MAX_ITERATIONS = 25

PRICE_INPUT_PER_MTOK_UC = 590.0
PRICE_OUTPUT_PER_MTOK_UC = 790.0

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

USER_PROMPT = (
    "Use the execute_sql tool to find all customers in the 'enterprise' "
    "tier whose contracts expire in Q1 2026. If the query fails, inspect "
    "the error message and retry with a corrected query."
)
TOOL_ERROR_MESSAGE = "ERROR: column 'tier' does not exist. Did you mean 'tier_name'?"


@dataclass
class RunRecord:
    run_id: str
    runtime: str
    provider: str
    workload: str
    iteration: int
    n_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_uc: float
    cap_uc: int
    overshoot_uc: float
    outcome: str
    wall_clock_s: float
    error: str = ""


def synth_cost_uc(in_tok: int, out_tok: int) -> float:
    return (in_tok * PRICE_INPUT_PER_MTOK_UC / 1_000_000.0
            + out_tok * PRICE_OUTPUT_PER_MTOK_UC / 1_000_000.0)


def check_prereqs():
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: pip install requests")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        if MODEL not in models:
            sys.exit(f"ERROR: model {MODEL} not found in Ollama. "
                     f"Run: ollama pull {MODEL}")
    except Exception as e:
        sys.exit(f"ERROR: cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
    print(f"OK: Ollama is up, {MODEL} is available.")


def run_langgraph(iteration: int, cap_uc: int) -> RunRecord:
    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from langchain_core.callbacks import BaseCallbackHandler

    @tool
    def execute_sql(query: str) -> str:
        return TOOL_ERROR_MESSAGE

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL, temperature=0.0)

    in_tok, out_tok, n_calls = 0, 0, 0

    class UsageRecorder(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            nonlocal in_tok, out_tok, n_calls
            n_calls += 1
            try:
                gen = response.generations[0][0]
                meta = gen.message.usage_metadata or {}
                in_tok += meta.get("input_tokens", 0)
                out_tok += meta.get("output_tokens", 0)
            except (AttributeError, IndexError, TypeError):
                pass

    recorder = UsageRecorder()
    agent = create_react_agent(llm, [execute_sql])

    t0 = time.time()
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": USER_PROMPT}]},
            config={"recursion_limit": RECURSION_LIMIT, "callbacks": [recorder]},
        )
        outcome = "completed"
    except Exception as e:
        msg = str(e).lower()
        outcome = "recursion_limit" if "recursion" in msg else "error"
    elapsed = time.time() - t0

    cost = synth_cost_uc(in_tok, out_tok)
    return RunRecord(
        run_id=f"langgraph_{iteration:02d}",
        runtime="langgraph", provider="ollama", workload="lang001",
        iteration=iteration, n_calls=n_calls,
        total_input_tokens=in_tok, total_output_tokens=out_tok,
        total_cost_uc=round(cost, 4), cap_uc=cap_uc,
        overshoot_uc=round(max(0.0, cost - cap_uc), 4),
        outcome=outcome, wall_clock_s=round(elapsed, 2),
    )


def run_langgraph_agentguard(iteration: int, cap_uc: int) -> RunRecord:
    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from langchain_core.callbacks import BaseCallbackHandler

    @tool
    def execute_sql(query: str) -> str:
        return TOOL_ERROR_MESSAGE

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL, temperature=0.0)

    in_tok, out_tok, n_calls = 0, 0, 0
    cap_hit = False

    class AgentGuard(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            nonlocal in_tok, out_tok, n_calls, cap_hit
            n_calls += 1
            try:
                gen = response.generations[0][0]
                meta = gen.message.usage_metadata or {}
                in_tok += meta.get("input_tokens", 0)
                out_tok += meta.get("output_tokens", 0)
            except (AttributeError, IndexError, TypeError):
                pass
            cum = synth_cost_uc(in_tok, out_tok)
            if cum > cap_uc and not cap_hit:
                cap_hit = True

    guard = AgentGuard()
    agent = create_react_agent(llm, [execute_sql])

    t0 = time.time()
    outcome = "completed"
    try:
        for _ in agent.stream(
                {"messages": [{"role": "user", "content": USER_PROMPT}]},
                config={"recursion_limit": RECURSION_LIMIT, "callbacks": [guard]},
        ):
            if cap_hit:
                outcome = "agentguard_fired"
                break
    except Exception as e:
        outcome = "recursion_limit" if "recursion" in str(e).lower() else "error"
    elapsed = time.time() - t0

    cost = synth_cost_uc(in_tok, out_tok)
    return RunRecord(
        run_id=f"langgraph_agentguard_{iteration:02d}",
        runtime="langgraph_agentguard", provider="ollama", workload="lang001",
        iteration=iteration, n_calls=n_calls,
        total_input_tokens=in_tok, total_output_tokens=out_tok,
        total_cost_uc=round(cost, 4), cap_uc=cap_uc,
        overshoot_uc=round(max(0.0, cost - cap_uc), 4),
        outcome=outcome, wall_clock_s=round(elapsed, 2),
    )


def run_token_budgets(iteration: int, cap_uc: int) -> RunRecord:
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool
    from langchain_core.callbacks import BaseCallbackHandler

    SAFETY_MARGIN = 1.0

    @tool
    def execute_sql(query: str) -> str:
        return TOOL_ERROR_MESSAGE

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL, temperature=0.0)

    in_tok, out_tok, n_calls = 0, 0, 0
    budget_remaining = float(cap_uc)
    outcome = "completed"

    class UsageRecorder(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            nonlocal in_tok, out_tok, n_calls
            n_calls += 1
            try:
                gen = response.generations[0][0]
                meta = gen.message.usage_metadata or {}
                in_tok += meta.get("input_tokens", 0)
                out_tok += meta.get("output_tokens", 0)
            except (AttributeError, IndexError, TypeError):
                pass

    recorder = UsageRecorder()

    messages = [{"role": "user", "content": USER_PROMPT}]
    tools = [execute_sql]

    t0 = time.time()
    for step in range(MAX_ITERATIONS):
        payload = json.dumps({
            "messages": [{"role": m.get("role", "user"),
                          "content": str(m.get("content", m))}
                         for m in messages],
            "tools": [{"name": t.name, "description": t.description} for t in tools],
        })
        bytes_in = len(payload.encode("utf-8"))
        est_in_tokens = math.ceil(bytes_in * SAFETY_MARGIN)
        max_output = 512
        est_call_cost = synth_cost_uc(est_in_tokens, max_output)

        if est_call_cost > budget_remaining:
            outcome = "tb_refused"
            break

        budget_remaining -= est_call_cost

        try:
            ai_msg = llm.bind_tools(tools).invoke(
                messages, config={"callbacks": [recorder]}
            )
        except Exception as e:
            outcome = "error"
            break

        try:
            meta = ai_msg.usage_metadata or {}
            actual_call_cost = synth_cost_uc(
                meta.get("input_tokens", est_in_tokens),
                meta.get("output_tokens", max_output),
            )
            refund = max(0.0, est_call_cost - actual_call_cost)
            budget_remaining += refund
        except (AttributeError, TypeError):
            pass

        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            outcome = "completed"
            break

        for tc in ai_msg.tool_calls:
            try:
                tool_result = execute_sql.invoke(tc["args"])
            except Exception:
                tool_result = TOOL_ERROR_MESSAGE
            messages.append({"role": "tool", "content": str(tool_result),
                             "tool_call_id": tc["id"]})

    elapsed = time.time() - t0
    cost = synth_cost_uc(in_tok, out_tok)
    return RunRecord(
        run_id=f"token_budgets_{iteration:02d}",
        runtime="token_budgets", provider="ollama", workload="lang001",
        iteration=iteration, n_calls=n_calls,
        total_input_tokens=in_tok, total_output_tokens=out_tok,
        total_cost_uc=round(cost, 4), cap_uc=cap_uc,
        overshoot_uc=round(max(0.0, cost - cap_uc), 4),
        outcome=outcome, wall_clock_s=round(elapsed, 2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-uc", type=int, default=100,
                        help="Cap in micro-cents (default 100, matches §5.7)")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    args = parser.parse_args()

    check_prereqs()

    print(f"\nOllama LANG-001 replication (CORRECTED v2)")
    print(f"  Model: {MODEL}    Cap: {args.cap_uc} uc    Runs: {args.n_runs} per runtime\n")

    all_records: List[RunRecord] = []
    for runtime_name, runner in [
        ("langgraph", run_langgraph),
        ("langgraph_agentguard", run_langgraph_agentguard),
        ("token_budgets", run_token_budgets),
    ]:
        print(f"=== {runtime_name} ===")
        for i in range(args.n_runs):
            print(f"  run {i+1}/{args.n_runs}...", end=" ", flush=True)
            try:
                rec = runner(i, args.cap_uc)
                print(f"{rec.outcome}  cost={rec.total_cost_uc:.2f}/{rec.cap_uc} uc  "
                      f"calls={rec.n_calls}  overshoot={rec.overshoot_uc:.2f}  "
                      f"[{rec.wall_clock_s:.1f}s]")
                all_records.append(rec)
            except Exception as e:
                print(f"FATAL: {type(e).__name__}: {e}")
                all_records.append(RunRecord(
                    run_id=f"{runtime_name}_{i:02d}",
                    runtime=runtime_name, provider="ollama", workload="lang001",
                    iteration=i, n_calls=0, total_input_tokens=0,
                    total_output_tokens=0, total_cost_uc=0.0,
                    cap_uc=args.cap_uc, overshoot_uc=0.0,
                    outcome="fatal", wall_clock_s=0.0, error=str(e),
                ))

    csv_path = OUTPUT_DIR / "ollama_lang001_n10.csv"
    with open(csv_path, "w", newline="") as f:
        if all_records:
            writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(all_records[0])])
            writer.writeheader()
            for r in all_records:
                writer.writerow(asdict(r))
    print(f"\nResults written to {csv_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for runtime_name in ["langgraph", "langgraph_agentguard", "token_budgets"]:
        runs = [r for r in all_records if r.runtime == runtime_name and not r.error]
        if not runs:
            print(f"  {runtime_name:>25}: no valid runs")
            continue
        mean_cost = sum(r.total_cost_uc for r in runs) / len(runs)
        max_cost = max(r.total_cost_uc for r in runs)
        n_overshoot = sum(1 for r in runs if r.overshoot_uc > 0)
        pct_cap = 100 * mean_cost / args.cap_uc
        print(f"  {runtime_name:>25}: mean={mean_cost:6.2f} uc  ({pct_cap:5.1f}% of cap)  "
              f"max={max_cost:6.2f}  overshoot {n_overshoot}/{len(runs)}")

    print("-" * 70)
    print("Ollama llama3.2:latest is a 3B-param model and typically abandons")
    print("retry loops after 2-3 calls (see §5.7 of the paper). If all three")
    print("runtimes show 'completed' at low cost, that is the honest finding:")
    print("the LANG-001 retry pattern doesn't reproduce on this model. The")


if __name__ == "__main__":
    main()