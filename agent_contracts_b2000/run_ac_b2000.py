from __future__ import annotations
import argparse, csv, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
import agent_contracts as ac

try:
    import litellm
    _HAVE_LITELLM = True
except ImportError:
    _HAVE_LITELLM = False

MODEL = "claude-haiku-4-5"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 200
MAX_AGENT_STEPS = 20

CAP_USD = 0.02
UC_PER_USD = 100_000

PRICE_IN_PER_MTOK_USD = 1.0
PRICE_OUT_PER_MTOK_USD = 5.0

MARGIN = 2.0

DEFAULT_PROMPTS = {
    "system": (
        "You are a senior data analyst agent operating against an "
        "e-commerce analytics database. Answer questions about the data "
        "by issuing SQL queries through the execute_sql tool. You have "
        "no prior knowledge of the data; all conclusions must come from "
        "query results.\n\n"
        "CORE RULES:\n"
        "1. Never answer in prose without running at least one query.\n"
        "2. If a query returns an error, examine the error carefully and "
        "retry with a corrected query. Do not give up after one attempt.\n"
        "3. Always include ORDER BY in aggregations for determinism.\n"
        "4. Use ANSI SQL with PostgreSQL extensions (CTEs, window "
        "functions, INTERVAL arithmetic, JSONB operators).\n"
        "5. Apply explicit type casts when comparing across types.\n"
        "6. Quote identifiers conflicting with reserved words.\n\n"
        "DATABASE SCHEMA:\n\n"
        "Table: users\n"
        "  id                BIGINT PRIMARY KEY\n"
        "  email             TEXT UNIQUE NOT NULL\n"
        "  password_hash     TEXT NOT NULL\n"
        "  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()\n"
        "  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()\n"
        "  country_code      CHAR(2) NOT NULL\n"
        "  signup_source     TEXT  -- web,ios,android,partner\n"
        "  subscription_tier TEXT NOT NULL DEFAULT 'free'\n"
        "                    -- free,standard,premium\n"
        "  last_login_at     TIMESTAMPTZ\n"
        "  email_verified    BOOLEAN NOT NULL DEFAULT FALSE\n"
        "  locale            TEXT DEFAULT 'en-US'\n\n"
        "Table: orders\n"
        "  id                 BIGINT PRIMARY KEY\n"
        "  user_id            BIGINT NOT NULL REFERENCES users(id)\n"
        "  placed_at          TIMESTAMPTZ NOT NULL\n"
        "  status             TEXT NOT NULL\n"
        "                     -- pending,paid,shipped,delivered,\n"
        "                     -- cancelled,refunded\n"
        "  total_amount_cents BIGINT NOT NULL\n"
        "  currency_code      CHAR(3) NOT NULL DEFAULT 'USD'\n"
        "  shipping_country   CHAR(2) NOT NULL\n"
        "  payment_method     TEXT NOT NULL\n"
        "                     -- card,paypal,wallet,invoice\n"
        "  coupon_code        TEXT\n"
        "  discount_cents     BIGINT NOT NULL DEFAULT 0\n"
        "  tax_cents          BIGINT NOT NULL DEFAULT 0\n"
        "  shipping_cents     BIGINT NOT NULL DEFAULT 0\n\n"
        "Table: products\n"
        "  id                BIGINT PRIMARY KEY\n"
        "  sku               TEXT UNIQUE NOT NULL\n"
        "  name              TEXT NOT NULL\n"
        "  list_price_cents  BIGINT NOT NULL\n"
        "  category          TEXT NOT NULL\n"
        "  supplier_id       BIGINT NOT NULL\n"
        "  inventory_count   INTEGER NOT NULL DEFAULT 0\n"
        "  is_active         BOOLEAN NOT NULL DEFAULT TRUE\n\n"
        "Table: order_items\n"
        "  order_id          BIGINT NOT NULL REFERENCES orders(id)\n"
        "  product_id        BIGINT NOT NULL REFERENCES products(id)\n"
        "  quantity          INTEGER NOT NULL\n"
        "  unit_price_cents  BIGINT NOT NULL\n"
        "  line_total_cents  BIGINT NOT NULL\n"
        "  PRIMARY KEY (order_id, product_id)\n\n"
        "Table: reviews\n"
        "  id                BIGINT PRIMARY KEY\n"
        "  product_id        BIGINT NOT NULL REFERENCES products(id)\n"
        "  user_id           BIGINT NOT NULL REFERENCES users(id)\n"
        "  rating            SMALLINT NOT NULL\n"
        "  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()\n\n"
        "Table: sessions\n"
        "  id                UUID PRIMARY KEY\n"
        "  user_id           BIGINT REFERENCES users(id)\n"
        "  started_at        TIMESTAMPTZ NOT NULL\n"
        "  ended_at          TIMESTAMPTZ\n\n"
        "DATE AND TIME HANDLING:\n"
        "For 'last 7 days' use: created_at >= NOW() - INTERVAL '7 days'\n"
        "For 'this month': DATE_TRUNC('month', created_at) "
        "= DATE_TRUNC('month', NOW())\n"
        "All timestamps are TIMESTAMPTZ in UTC.\n\n"
        "EXAMPLE QUERY PATTERN:\n\n"
        "Daily aggregation with status filter:\n"
        "  SELECT DATE_TRUNC('day', placed_at)::date AS day,\n"
        "         COUNT(*) AS orders,\n"
        "         SUM(total_amount_cents)::numeric/100 AS revenue\n"
        "  FROM orders\n"
        "  WHERE placed_at >= NOW() - INTERVAL '30 days'\n"
        "    AND status NOT IN ('cancelled','refunded')\n"
        "  GROUP BY 1 ORDER BY 1\n\n"
        "Indexed columns (use these for WHERE/JOIN):\n"
        "  users(email), users(created_at), users(country_code,created_at)\n"
        "  orders(user_id,placed_at), orders(placed_at), orders(status)\n"
    ),
    "user": (
        "For our weekly executive report: how many distinct users signed "
        "up in the last 7 days, broken down by their country code and "
        "subscription tier? Sort by signup count descending, ties broken "
        "alphabetically by country code. Exclude unverified email "
        "accounts (email_verified = false) from the count."
    ),
    "tools": [
        {
            "name": "execute_sql",
            "description": (
                "Execute a read-only SQL query against the application "
                "analytics database (PostgreSQL 16) and return rows as "
                "JSON. Returns an error if the SQL is invalid, "
                "references unknown columns, or exceeds the 30-second "
                "per-query timeout. Only SELECT statements are permitted."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A single ANSI SQL SELECT statement. "
                            "PostgreSQL extensions (CTEs, window "
                            "functions, INTERVAL, JSONB) are supported."
                        ),
                    },
                },
                "required": ["query"],
            },
        }
    ],
    "tool_error": (
        "Error: column 'signup_date' does not exist on relation 'users'. "
        "Did you mean 'created_at'? The users table uses 'created_at' "
        "for the sign-up timestamp — re-read the schema and retry."
    ),
}

def estimate_input_byte_length(messages: list[dict], tools: list[dict],
                               system: str) -> int:
    payload = {"system": system, "messages": messages, "tools": tools}
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

def estimate_call_cost_usd(messages: list[dict], tools: list[dict],
                           system: str) -> float:
    in_tokens = estimate_input_byte_length(messages, tools, system)
    in_cost = MARGIN * in_tokens * PRICE_IN_PER_MTOK_USD / 1_000_000
    out_cost = MAX_OUTPUT_TOKENS * PRICE_OUT_PER_MTOK_USD / 1_000_000
    return in_cost + out_cost

@dataclass
class TrialResult:
    trial_id: int
    outcome: str
    steps_admitted: int
    cumulative_spend_uc: int
    cap_uc: int
    final_remaining_uc: int
    pre_flight_refusals: int
    refusal_reservation_uc: int
    elapsed_s: float
    error: str = ""

def run_one_trial(trial_id: int, prompts: dict, dry_run: bool = False,
                  verbose: bool = False) -> TrialResult:
    t0 = time.time()

    contract = ac.Contract(
        id=f"tb-headtohead-trial-{trial_id}",
        name="LANG-001 retry loop under cumulative cost cap",
        resources=ac.ResourceConstraints(cost_usd=CAP_USD),
        mode=ac.ContractMode.BALANCED,
    )
    monitor = ac.ResourceMonitor(constraints=contract.resources)

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
        remaining_usd = monitor.get_remaining_cost()
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
                usage = resp.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
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
                                   int(round(monitor.usage.cost_usd * UC_PER_USD)),
                                   int(CAP_USD * UC_PER_USD),
                                   int(round(monitor.get_remaining_cost() * UC_PER_USD)),
                                   pre_flight_refusals, refusal_reservation_uc,
                                   time.time() - t0, err_str)

        actual_usd = (input_tokens * PRICE_IN_PER_MTOK_USD / 1_000_000
                      + output_tokens * PRICE_OUT_PER_MTOK_USD / 1_000_000)
        monitor.usage.cost_usd += actual_usd
        monitor.usage.api_calls += 1
        monitor.usage.tokens += input_tokens + output_tokens
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

        if monitor.is_violated():
            outcome = "violation"
            break
    else:
        outcome = "step_limit"

    return TrialResult(
        trial_id=trial_id,
        outcome=outcome,
        steps_admitted=steps_admitted,
        cumulative_spend_uc=int(round(monitor.usage.cost_usd * UC_PER_USD)),
        cap_uc=int(CAP_USD * UC_PER_USD),
        final_remaining_uc=int(round(monitor.get_remaining_cost() * UC_PER_USD)),
        pre_flight_refusals=pre_flight_refusals,
        refusal_reservation_uc=refusal_reservation_uc,
        elapsed_s=time.time() - t0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--output", type=str, default="ac_b2000_results.csv")
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
        print("ERROR: ANTHROPIC_API_KEY not set. Use --dry-run for offline test.",
              file=sys.stderr)
        sys.exit(1)

    init_messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]},
    ]
    init_bytes = estimate_input_byte_length(init_messages, prompts["tools"],
                                            prompts["system"])
    init_estimate_uc = int(round(estimate_call_cost_usd(
        init_messages, prompts["tools"], prompts["system"]) * UC_PER_USD))

    cap_uc = int(CAP_USD * UC_PER_USD)
    print(f"Agent Contracts head-to-head, B_0 = {cap_uc} uc (${CAP_USD:.4f})")
    print(f"  Model: {MODEL}, T={TEMPERATURE}, max_output={MAX_OUTPUT_TOKENS}")
    print(f"  ai-agent-contracts: {getattr(ac, '__version__', 'unknown')}")
    print(f"  Initial payload: {init_bytes} bytes, "
          f"step 0 estimate: {init_estimate_uc} uc")
    print(f"  Trials: {args.trials}, dry-run: {args.dry_run}")
    print(f"  Output: {args.output}")
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

    from collections import Counter
    outcomes = Counter(r.outcome for r in results)
    print("\nSummary")
    for k, v in outcomes.most_common():
        print(f"  {k}: {v}/{len(results)}")
    overshoots = sum(1 for r in results if r.cumulative_spend_uc > r.cap_uc)
    print(f"  overshoots (spend > cap): {overshoots}/{len(results)}")


if __name__ == "__main__":
    main()