# token-budgets-experiments

Experimental harnesses and result CSVs for the *Token Budgets* paper (preprint,
2026). This repository holds the data and code behind the paper's quantitative
claims. All result CSVs are shipped in-repo; no API calls are required to
reproduce the analyses.

## Structure

```
.
├── agent_contracts_b2000/   # Agent Contracts discriminating-cap head-to-head (Table 9)
├── budget-spike/            # LANG-001 retry-loop reproduction + multi-agent harness
├── experiments/             # anthropic_estimator (A1), a7_fault_injection, eval_at_scale
├── fair-baseline/           # Independent baseline cohort analysis (paper §2.3)
├── forgetful_operator/      # Forgetful-Operator experiment incl. Condition E (Table 15)
├── governor-bench/          # Microbench for the governor-runtime extension
├── multiway/                # Multi-runtime sweep result CSVs (sweep_results/)
├── refund-live/             # Live refund + A7 fault-injection data (§5.27)
├── tools/                   # Sweep drivers, incl. multiway_compare.py
├── LICENSE-APACHE  LICENSE-MIT  README.md
```

## Headline results (shipped CSVs, no API access required)

| Result                                          | File                                                                 | Paper ref |
|-------------------------------------------------|----------------------------------------------------------------------|-----------|
| Five-runtime + Agent Contracts on gpt-4o (N=30) | `multiway/sweep_results/gpt4o_lang001_n30_full.csv`                  | Table 5   |
| Cross-provider replication on Anthropic (N=30)  | `multiway/sweep_results/claude_sonnet_lang001_n30_full.csv`         | Table 6   |
| Discriminating cap B0=2000, three-way (N=30)    | `agent_contracts_b2000/results/`                                     | Table 9   |
| A1 validation (30/30 at margin 2.0)             | `experiments/anthropic_estimator/results/a1_validation.json`         | §5.30     |
| refund-live 1000-session sweep                  | `refund-live/refund_live_1000_results.csv`                           | §5.27 (A7)|
| Forgetful-Operator Condition E (0/30)           | `forgetful_operator/condition_e_rust_shared/condition_e_results.csv` | Table 15  |

## Table 5 reconciliation (no API calls)

```bash
python3 - <<'PY'
import csv
from collections import defaultdict
rts = defaultdict(lambda: {"n":0, "ov":0})
with open("multiway/sweep_results/gpt4o_lang001_n30_full.csv") as f:
    for r in csv.DictReader(f):
        rt = r["runtime"]; rts[rt]["n"] += 1
        if int(r.get("overshoot_uc","0") or 0) > 0: rts[rt]["ov"] += 1
for rt, s in sorted(rts.items()):
    print(f"  {rt:30s} {s['ov']}/{s['n']}")
PY
```

Expected: the five production baselines (LangGraph, CrewAI, AutoGen,
AgentGuard-callback, LiteLLM-proxy) overshoot 30/30 on LANG-001; Token Budgets
overshoots 0/30. Agent Contracts (Ye & Tan, COINE 2026) also reaches 0/N via
runtime monitoring; the affine layer adds non-bypassability of in-program
integrity (no cloning, double-spend, or post-delegation use) that runtime
monitoring cannot enforce — isolated by the Forgetful-Operator experiment
(`forgetful_operator/`, Table 15).

## A1 validation

```bash
cd experiments/anthropic_estimator
python3 a1_runner.py --margin 2.0   # A1 holds 30/30 cells at margin 2.0
python3 a1_runner.py --margin 1.0   # A1 holds 1/3 cells (margin is load-bearing)
```

## A7 fault injection (provider under-reporting)

```bash
cd experiments
python3 a7_fault_injection.py --cap 2000 --trials 1000 \
  --cost-csv ../refund-live/refund_live_1000_results.csv \
  --cost-col actual_uc --reservation-col reservation_uc \
  --k 1.0 2.0 5.0 10.0
```

k=1 (truthful provider) -> 0/1000 overshoot (Lemma 1 holds); k=5, k=10 ->
1000/1000 (the A7 trust boundary). Reproduces the paper's A7 table.

## Multi-runtime live re-run (optional, ~$0.50-$1.00)

```bash
export OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-...
python3 tools/multiway_compare.py --provider openai --workload lang001 --runs 30 \
        --output-csv my_rerun.csv
```

## Known issues

- `tools/multiway_compare.py` depends on `ai-agent-contracts` being installed and
  patched per the in-repo install note; shipped CSVs were generated against
  `ai-agent-contracts v0.3.1`.
- Live-API re-runs are subject to model nondeterminism; expect +/-1/30 variation
  on overshoot rates where the runtime cannot be fully pinned.

## License

Dual MIT/Apache-2.0.