# token-budgets-experiments

Experimental harnesses and result CSVs for the *Token Budgets* paper, currently under review at *Empirical Software Engineering*.

This repository contains the data and code for every quantitative claim in the paper. All result CSVs are shipped in-repo; no API calls are required to reproduce the analyses.

## Structure

```
.
├── budget-spike/                  # Reproducible LANG-001 retry-loop incident
├── fair-baseline/                 # Fair-baseline corpus and selection methodology
├── governor-bench/                # Microbench for the governor-runtime extension
├── multiway/                      # 6-runtime head-to-head on gpt-4o (Table 30)
├── refund-live/                   # Live refund evidence (§5.24)
├── experiments/
│   ├── anthropic_estimator/       # A1 calibration (margin 2.0 holds 30/30)
│   └── conjecture_1_stress/       # 10,000-iteration stress sweep (Conjecture 1)
└── README.md
```

## Headline results (shipped CSVs, no API access required)

| Result                                            | File                                                                          | Paper reference |
|---------------------------------------------------|-------------------------------------------------------------------------------|-----------------|
| 6-runtime head-to-head on gpt-4o                  | `multiway/sweep_results/gpt4o_lang001_n10_full.csv`                          | Table 30        |
| Agent Contracts head-to-head                      | `multiway/sweep_results/agent_contracts_lang001_n10.csv`                     | Table 30 row 7  |
| A1 validation (30/30 holds at margin 2.0)         | `experiments/anthropic_estimator/results/runs.csv`                           | §5.22           |
| A1 summary                                        | `experiments/anthropic_estimator/results/a1_validation.json`                 | §5.22           |
| refund-live core (10 sessions)                    | `refund-live/refund_live_results.csv`                                        | §5.24           |
| refund-live 1000-session sweep                    | `refund-live/refund_live_1000_results.csv`                                   | §5.24           |
| Conjecture 1 stress (10,000 iterations)           | `experiments/conjecture_1_stress/results/iterations.csv`                     | §IV-E (open)    |
| Fair-baseline corpus                              | `fair-baseline/fair_baseline_results.csv`                                    | §5.7            |

## Table 30 reconciliation (no API calls)

```bash
python3 - <<'PY'
import csv
from collections import defaultdict
rts = defaultdict(lambda: {"n":0, "ov":0})
with open("multiway/sweep_results/gpt4o_lang001_n10_full.csv") as f:
    for r in csv.DictReader(f):
        rt = r["runtime"]
        rts[rt]["n"] += 1
        if int(r.get("overshoot_uc","0") or 0) > 0:
            rts[rt]["ov"] += 1
for rt, s in sorted(rts.items()):
    marker = " ← 0/10" if s["ov"] == 0 else ""
    print(f"  {rt:30s} {s['ov']}/{s['n']}{marker}")
PY
```

Expected output:

```
  autogen                        10/10
  crewai                         10/10
  langgraph_only                 10/10
  langgraph_with_guard           10/10
  litellm_proxy                  10/10
  token_capabilities              0/10 ← 0/10
```

Five production frameworks overshoot the user cap on every run of the same LANG-001 workload on OpenAI gpt-4o. Token Capabilities (this paper's contribution) prevents the overshoot at compile time. Agent Contracts (Ye & Tan, COINE 2026, see `agent_contracts_lang001_n10.csv`) also achieves 0/10 via runtime monitoring; the affine integrity layer in Token Capabilities provides additional guarantees against budget cloning, double-spending, and post-delegation use that pure runtime monitoring cannot enforce.

## Live re-run (optional, costs ~$1)

To re-execute the multi-runtime sweep against the live OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
cd multiway
uv sync   # or: pip install -r requirements.txt
python3 multiway_compare.py --provider openai --workload lang001 --runs 10 \
        --output-csv my_rerun.csv
```

The sweep takes ~10 minutes wall-clock and ~$0.50–$1.00 in OpenAI charges. Compare your CSV against the shipped `gpt4o_lang001_n10_full.csv` — overshoot rates per runtime should be within ±1/10 due to model nondeterminism.

## A1 validation reproduction

```bash
cd experiments/anthropic_estimator
python3 a1_runner.py --margin 2.0
# Should print: A1 holds 30/30 cells at margin 2.0
python3 a1_runner.py --margin 1.0
# Should print: A1 holds 1/3 cells at margin 1.0 (margin is load-bearing)
```

The detailed results are in `results/runs.csv`; the JSON summary at `results/a1_validation.json` reproduces the numbers reported in §5.22.

## Conjecture 1 stress sweep

```bash
cd experiments/conjecture_1_stress
python3 stress_runner.py --iterations 10000 --output results/my_iterations.csv
# Compare against shipped results/iterations.csv
```

10,000 randomized iterations of the abstract semantics, every iteration preserving the cap-soundness invariant. This is supporting empirical evidence; the proof obligation itself remains open (Conjecture 1, §IV-E).

## Refund-live

```bash
cd refund-live
python3 refund_live.py --sessions 10   # reproduces refund_live_results.csv
python3 refund_live.py --sessions 1000 # reproduces refund_live_1000_results.csv (~$3)
```

Live refunds against Anthropic Claude with reservation-and-actual-spend reconciliation. The 1000-session sweep shows refund accuracy converges to within ±2% across all session lengths.

## Known issues

- The `multiway/multiway_compare.py` runner depends on the upstream `agent-contracts` package being installed and patched per `multiway/AGENT_CONTRACTS_INSTALL.md`. The shipped CSVs were generated against `ai-agent-contracts v0.3.1`.
- All live-API re-runs are subject to model nondeterminism. The CSVs ship deterministic seeds where the runtime supports them, but OpenAI gpt-4o cannot be fully pinned. Expect ±1/10 variation on overshoot rates.

## Companion repositories

- [token-budgets](https://github.com/sajjadanwar0/token-budgets) — main library
- [token-budgets-formals](https://github.com/sajjadanwar0/token-budgets-formals) — formal verification + IRR
- [token-budgets-baseline](https://github.com/sajjadanwar0/token-budgets-baseline) — fair-baseline corpus
- [token-budgets-python](https://github.com/sajjadanwar0/token-budgets-python) — Python port
- [token-budgets-extensions](https://github.com/sajjadanwar0/token-budgets-extensions) — adaptive estimator + future work

## License

Dual MIT/Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.