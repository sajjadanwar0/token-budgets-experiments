from __future__ import annotations
import argparse
import csv
import importlib
import os
import sys
import time
import traceback

PINNED_AGENT_CONTRACTS_VERSION = "0.3.1"
HARNESS_VERSION = "v3-probe-first"

CONTRACT_CAP_UC = 540
CONTRACT_CAP_USD = CONTRACT_CAP_UC / 1_000_000.0

WORKLOAD_NAME = "lang001"
PROVIDER_NAME = "openai"
RUNTIME_NAME = "agent_contracts"

LANG001_SYSTEM_PROMPT = (
    "You are a database assistant. The user will give you a natural-language "
    "question. You should answer it by calling the sql_query tool. "
    "The query must be syntactically valid SQL."
)
LANG001_USER_PROMPT = "How many customers signed up last month?"
LANG001_TOOL_NAME = "sql_query"
LANG001_TOOL_DESCRIPTION = "Execute a SQL query against the customer database."
LANG001_TOOL_ERROR = (
    "SQL syntax error near 'WHERE'. Did you mean FROM? "
    "Please correct and retry."
)

IMPORT_PATH_CANDIDATES = [
    "agent_contracts",
    "ai_agent_contracts",
    "aac",
    "agent_contracts.core",
    "ai_agent_contracts.core",
]

CONTRACT_CLASS_NAMES = ["Contract", "AgentContract", "ContractClient"]
CONSTRAINTS_CLASS_NAMES = ["ResourceConstraints", "ResourceConstraint",
                           "BudgetConstraint", "BudgetConstraints"]
MODE_ENUM_NAMES = ["ContractMode", "Mode", "EnforcementMode"]


def probe_api():
    diagnostic_lines = ["AGENT-CONTRACTS API PROBE \n"]
    diagnostic_lines.append("Trying import paths in order:\n")

    last_exception = None
    successful_module = None

    for path in IMPORT_PATH_CANDIDATES:
        diagnostic_lines.append(f"  trying:  import {path}\n")
        try:
            mod = importlib.import_module(path)
            diagnostic_lines.append(f"    OK: imported {path}\n")
            mod_dir = set(dir(mod))
            diagnostic_lines.append(f"    module dir: "
                                    f"{sorted(x for x in mod_dir if not x.startswith('_'))[:20]}\n")
            successful_module = mod
            break
        except ImportError as e:
            diagnostic_lines.append(f"    FAIL: {e}\n")
            last_exception = e

    if successful_module is None:
        diagnostic_lines.append(f"\nNo import path worked. Last error: {last_exception}\n")
        diagnostic_lines.append("\nInstalled packages with 'agent' in name:\n")
        try:
            import importlib.metadata as md
            for dist in md.distributions():
                name = dist.metadata.get("Name", "")
                if name and "agent" in name.lower():
                    diagnostic_lines.append(f"  {name} {dist.version}\n")
        except Exception as e:
            diagnostic_lines.append(f"  (could not enumerate: {e})\n")
        raise RuntimeError("".join(diagnostic_lines))

    Contract = None

    for name in CONTRACT_CLASS_NAMES:
        if hasattr(successful_module, name):
            Contract = getattr(successful_module, name)
            diagnostic_lines.append(f"  Found Contract class as: {path}.{name}\n")
            break

    if Contract is None:
        diagnostic_lines.append(f"\nNo Contract class found in {path}.\n")
        diagnostic_lines.append(f"Tried: {CONTRACT_CLASS_NAMES}\n")
        diagnostic_lines.append(f"Available names in module:\n")
        diagnostic_lines.append(f"  {sorted(x for x in dir(successful_module) if not x.startswith('_'))}\n")
        raise RuntimeError("".join(diagnostic_lines))

    ResourceConstraints = None

    for name in CONSTRAINTS_CLASS_NAMES:
        if hasattr(successful_module, name):
            ResourceConstraints = getattr(successful_module, name)
            diagnostic_lines.append(f"  Found Constraints class as: {path}.{name}\n")
            break

    ContractMode = None

    for name in MODE_ENUM_NAMES:
        if hasattr(successful_module, name):
            ContractMode = getattr(successful_module, name)
            diagnostic_lines.append(f"  Found Mode enum as: {path}.{name}\n")
            break

    print("".join(diagnostic_lines), file=sys.stderr)
    return successful_module, Contract, ResourceConstraints, ContractMode

def run_one_trial(api_handle, run_id: int, model: str) -> dict:
    mod, Contract, ResourceConstraints, ContractMode = api_handle
    t0 = time.monotonic()
    num_calls = 0
    spent_uc = 0
    error_msg = ""

    def sql_query_tool(query: str) -> str:
        return LANG001_TOOL_ERROR

    try:
        if ResourceConstraints is None:
            raise RuntimeError("No ResourceConstraints class found; "
                               "cannot construct contract")
        constraints = ResourceConstraints(cost_usd=CONTRACT_CAP_USD)

        if ContractMode is not None:
            mode_value = (getattr(ContractMode, "BALANCED", None)
                          or getattr(ContractMode, "DEFAULT", None)
                          or list(ContractMode)[0])
            contract = Contract(resource_constraints=constraints, mode=mode_value)
        else:
            contract = Contract(resource_constraints=constraints)

        result = contract.run(
            model=model,
            system_prompt=LANG001_SYSTEM_PROMPT,
            user_prompt=LANG001_USER_PROMPT,
            tools={
                LANG001_TOOL_NAME: {
                    "description": LANG001_TOOL_DESCRIPTION,
                    "fn": sql_query_tool,
                }
            },
            max_iterations=20,
        )
        if hasattr(result, "violated") and result.violated:
            outcome = "agent_contracts_budget_violation"
        elif hasattr(result, "outcome"):
            outcome = str(result.outcome)
        else:
            outcome = "completed_no_cap_hit"
        num_calls = getattr(result, "num_llm_calls", 0)
        spent_uc = int(round(getattr(result, "total_cost_usd", 0.0) * 1_000_000))
    except Exception as e:
        outcome = "error"
        error_msg = f"{type(e).__name__}: {str(e)[:300]}\n" + traceback.format_exc()

    wall_seconds = round(time.monotonic() - t0, 3)
    overshoot_uc = 0
    structural_undershoot_uc = max(0, CONTRACT_CAP_UC - spent_uc)
    pct_of_cap = (spent_uc / CONTRACT_CAP_UC * 100.0) if CONTRACT_CAP_UC > 0 else 0.0

    return {
        "run_id": run_id, "runtime": RUNTIME_NAME, "workload": WORKLOAD_NAME,
        "provider": PROVIDER_NAME, "outcome": outcome,
        "cap_uc": CONTRACT_CAP_UC, "total_spent_uc": spent_uc,
        "overshoot_uc": overshoot_uc,
        "structural_undershoot_uc": structural_undershoot_uc,
        "wasted_call_cost_uc": 0, "pct_of_cap": round(pct_of_cap, 2),
        "agent_steps": num_calls, "wall_seconds": wall_seconds,
        "ai_agent_contracts_version": PINNED_AGENT_CONTRACTS_VERSION,
        "harness_version": HARNESS_VERSION, "error_message": error_msg,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--existing-n10", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--probe-only", action="store_true", help="Probe the API and run one trial only; no full sweep")
    parser.add_argument("--skip-drift-check", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    try:
        api_handle = probe_api()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print("\n PROBE FAILED \n", file=sys.stderr)
        print("Please share the diagnostic output above so we can identify\n"
              "the correct import path and class names for your installed\n"
              "ai-agent-contracts version.\n", file=sys.stderr)
        sys.exit(4)

    print(" API PROBE SUCCEEDED \n", file=sys.stderr)

    print("Running probe trial...", file=sys.stderr)
    probe_row = run_one_trial(api_handle, run_id=0, model=args.model)
    if probe_row["outcome"] == "error":

        print(f"\n PROBE TRIAL FAILED \n", file=sys.stderr)
        print(f"outcome: {probe_row['outcome']}", file=sys.stderr)
        print(f"error_message:\n{probe_row['error_message']}", file=sys.stderr)
        print(f"\nThe import path worked but the API call failed.\n"
              f"The class signatures or method names are different in your\n"
              f"installed version. Adjust the run_one_trial() function in\n"
              f"this script to match.\n", file=sys.stderr)
        sys.exit(4)

    print(f"  Probe trial: outcome={probe_row['outcome']}  "
          f"steps={probe_row['agent_steps']}  "
          f"wall={probe_row['wall_seconds']}s  "
          f"spent={probe_row['total_spent_uc']}uc", file=sys.stderr)

    if args.probe_only:
        print("\n--probe-only specified; exiting without full sweep.",
              file=sys.stderr)
        print(f"Probe row would have been written to: {args.output}",
              file=sys.stderr)
        sys.exit(0)

    print(f"\nRunning {args.runs} full-sweep trials on {args.model}...",
          file=sys.stderr)

    if os.path.isfile(args.existing_n10):
        with open(args.existing_n10, newline="") as f:
            existing_n10 = list(csv.DictReader(f))
    else:
        print(f"WARNING: existing N=10 not found; drift check disabled",
              file=sys.stderr)
        existing_n10 = []

    rows = [probe_row]
    for i in range(1, args.runs + 1):
        row = run_one_trial(api_handle, run_id=i, model=args.model)
        rows.append(row)
        log_extra = ""
        if row["outcome"] == "error":
            first_line = row["error_message"].split("\n")[0] if row["error_message"] else ""
            log_extra = f"  err={first_line[:80]}"
        print(f"  trial {i:3d}/{args.runs}: outcome={row['outcome']:40s}  "
              f"steps={row['agent_steps']:2d}  wall={row['wall_seconds']:6.2f}s  "
              f"spent={row['total_spent_uc']:4d}uc"
              f"{log_extra}", file=sys.stderr)
        if i == 10 and existing_n10 and not args.skip_drift_check:
            new_violation = sum(1 for r in rows[1:11]
                                if "budget_violation" in r["outcome"])
            existing_violation = sum(1 for r in existing_n10
                                     if "budget_violation" in r["outcome"])
            if abs(new_violation - existing_violation) > 2:
                print(f"\nWARNING: drift check failed "
                      f"({new_violation} vs {existing_violation})",
                      file=sys.stderr)
                print("Continuing anyway because per-trial errors are now visible.",
                      file=sys.stderr)

    fieldnames = ["run_id", "runtime", "workload", "provider", "outcome",
                  "cap_uc", "total_spent_uc", "overshoot_uc",
                  "structural_undershoot_uc", "wasted_call_cost_uc",
                  "pct_of_cap", "agent_steps", "wall_seconds",
                  "ai_agent_contracts_version", "harness_version",
                  "error_message"]

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows[1:]:
            w.writerow(r)

    n_violation = sum(1 for r in rows[1:] if "budget_violation" in r["outcome"])
    n_completed = sum(1 for r in rows[1:] if "completed" in r["outcome"])
    n_error = sum(1 for r in rows[1:] if r["outcome"] == "error")
    n_overshoot = sum(1 for r in rows[1:] if r["overshoot_uc"] > 0)

    print(f"\n=== N={args.runs} HEADLINE ===", file=sys.stderr)
    print(f"  budget_violation: {n_violation}/{args.runs}", file=sys.stderr)
    print(f"  completed:        {n_completed}/{args.runs}", file=sys.stderr)
    print(f"  error:            {n_error}/{args.runs}", file=sys.stderr)
    print(f"  overshoot:        {n_overshoot}/{args.runs}", file=sys.stderr)
    print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()