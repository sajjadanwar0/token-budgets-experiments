# Token Budgets — Experiments

> Empirical validation harness for the [`token-budgets`](https://github.com/sajjadanwar0/token-budgets)
> affine-resource discipline. **5 independent sub-harnesses** producing
> the headline numbers in the paper.

[![Live calls](https://img.shields.io/badge/live_API_row--events-5%2C424-brightgreen)](#refund-live-the-live-api-validation-corpus)
[![Violations](https://img.shields.io/badge/cap--soundness_violations-0-brightgreen)](#refund-live-the-live-api-validation-corpus)
[![Over-reservation](https://img.shields.io/badge/mean_margin-6.20×-blue)](#refund-live-the-live-api-validation-corpus)
[![Catalog](https://img.shields.io/badge/catalog-167_rows_/_23_frameworks-blue)](#budget-spike-contention-sweeps--the-catalog)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue)](LICENSE-MIT)

This repository is the **research-grade empirical artifact**
accompanying the Token Budgets paper. It consolidates 5 sub-harnesses
that produce the validation evidence in paper §V.

## Sub-harnesses

| Sub-harness | Language | What it validates |
|---|---|---|
| **`refund-live/`** | Rust | The receipt/refund cycle against live LLM provider APIs (Anthropic, OpenAI, Gemini, Groq, Ollama). 10 binaries, **5,424 row-events** total. Primary empirical evidence. |
| **`budget-spike/`** | Rust | Contention sweeps + boundary tests. Includes `budget-archaeology.csv` (**167 entries across 23 frameworks**). |
| **`fair-baseline/`** | Rust | Pre-call (Token Budgets) vs post-call (Callback) cost-control comparison. Demonstrates TB has 0% overshoot vs Callback's 2-26%. |
| **`governor-bench/`** | Rust | Criterion microbenchmark vs the `governor` rate-limiter crate. Construction time, memory footprint, per-spend latency. |
| **`multiway/`** | Python | Multi-framework comparison: LangChain, LiteLLM, AutoGen. Demonstrates framework-agnostic applicability. |

Each sub-harness has its own `Cargo.toml` (Rust) or `pyproject.toml`
(Python) and is built independently.

## refund-live: the live-API validation corpus

**Headline result**: 5,424 row-events across 15 CSV result files,
**zero cap-soundness violations** observed in the violation-tracked
subsets (1,000 + 1,190 = 2,190 events with explicit violation
tracking).

### The 10 binaries

```
refund-live/src/bin/
├── a1-adversarial.rs           # A1-stress: prompts crafted to maximize tokens-per-byte
├── a1-adversarial-n100.rs      # Same, n=100 replication (→ 1,000 rows)
├── a1-rerun.rs                 # Reproducibility re-run of the A1 corpus
├── compare-estimators.rs       # ByteLength vs Tiktoken estimator comparison
├── compare-estimators-openai.rs # Same, with OpenAI tokenizer
├── multi-turn-session.rs        # Multi-turn conversation budget threading (1,190 turns)
├── react-agent-bench.rs         # ReAct-style tool-loop benchmark
├── reasoning-eval.rs            # DeepSeek-R1 / o3-mini reasoning-model evaluation
├── refund-live-1000.rs          # The primary 1,000-call sweep
└── refund-live-multi.rs         # Multi-provider sweep
```

### Verified row counts

| Result file | Rows | Notes |
|---|---:|---|
| `multi_turn_turns.csv` | 1,190 | 100 sessions × ~12 turns avg |
| `refund_live_1000_results.csv` | 1,000 | Primary single-call sweep |
| `a1_adversarial_n100_results.csv` | 1,000 | A1-stress, 0 violations observed |
| `refund_live_anthropic_haiku_4_5_*.csv` (×3) | 300 each | Haiku at varying max_tokens |
| `refund_live_openai_gpt_4o_mini_1024_300.csv` | 300 | OpenAI sweep |
| `refund_live_ollama_llama3_2_1024_300.csv` | 300 | Local Ollama sweep |
| `compare_estimators*.csv` (×2) | 200 each | ByteLength vs Tiktoken |
| `reasoning_o3_mini_4096_100.csv` | 100 | Reasoning model |
| `multi_turn_sessions.csv` | 100 | Session-level summary |
| `a1_rerun_results.csv` | 60 | Reproducibility re-run |
| `react_agent_results.csv` | 50 | Agent loop |
| `refund_live_results.csv` | 10 | Smoke test |
| **TOTAL** | **5,424** | Across 15 result files |

### Run individual binaries

```bash
cd refund-live/
export ANTHROPIC_API_KEY="..."   # or OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
cargo run --release --bin refund-live-1000 -- --help
```

The `--features tiktoken` flag enables the Tiktoken estimator
(otherwise byte-length is used, which is the paper-default A1 sound
upper bound).

### What we observe

| Property | Result |
|---|---|
| Cap-soundness violations in violation-tracked subsets | **0** (out of 2,190) |
| Receipt cycle exact reconciliation | 100% (no orphaned receipts) |
| Aggregate mean margin_ratio (over-reservation factor) | **6.20×** (n=5,190 valid samples) |
| Margin_ratio range | 1.02× to 131.43× |
| Refund slack recovered | ~84% of reservation on average |

### Honest scope

- **5,424 row-events is a corpus, not a proof.** A sufficiently
  adversarial prompt against an unbounded estimator could in
  principle exceed A1; we did not observe it within our corpus.
  The mechanized soundness of the receipt cycle across five
  independent provers (in
  [`token-budgets-formals/`](https://github.com/sajjadanwar0/token-budgets-formals))
  is the actual structural guarantee.
- **Two events ≠ two calls.** `multi_turn_turns.csv` rows are
  individual turns within multi-turn sessions; each session may
  have several turns. If you prefer to count sessions (100) rather
  than turns (1,190), the corpus size becomes 5,424 - 1,190 + 100
  = 4,334 calls. We report the larger number because each turn is
  a distinct provider exchange that goes through the budget
  arithmetic.
- **High refusal rate at tight caps.** On the adversarial corpus
  with tight per-call caps, a substantial fraction of prompts are
  rejected at the reservation step because the byte-length upper
  bound exceeds the cap. This is the cost of A1's universal upper
  bound and motivates the `AdaptiveEstimator` in
  [`token-budgets-extensions`](https://github.com/sajjadanwar0/token-budgets-extensions).

## budget-spike: contention sweeps + the catalog

`budget-spike/` contains:

- **Contention sweep**: parallel split/spend/merge across N tasks
  (N ∈ {4, 8, 16, 32, 64}). Output: `sweep_results/*/`.
- **Boundary tests**: cap-soundness at the const-generic boundaries.
- **`budget-archaeology.csv`**: The **167-entry failure catalog**
  across **23 frameworks and 18 ecosystems**, spanning 2023–2026.
  The codebook lives at
  [`token-budgets/data/budget-archaeology-codebook.md`](https://github.com/sajjadanwar0/token-budgets/blob/master/data/budget-archaeology-codebook.md).

Run with:

```bash
cd budget-spike/
cargo run --release --bin sweep -- --n 32
```

## fair-baseline: pre-call vs post-call comparison

`fair-baseline/` compares two cost-control strategies and produces
the data used in the paper's TB-vs-Callback comparison.

**Verified results from `fair_baseline_results.csv`**:

| Cap (uc) | Token Budgets overshoot | Callback baseline overshoot |
|---:|---:|---:|
| 5,000 | **0.00%** ✅ | 22.56% ❌ |
| 6,500 | **0.00%** ✅ | 25.75% ❌ |
| 8,000 | **0.00%** ✅ | 2.18% ❌ |
| 10,000 | **0.00%** ✅ | 2.07% ❌ |

Token Budgets achieves 0% overshoot at all four cap settings;
Callback-based budgeting overshoots by 2-26% depending on cap.

```bash
cd fair-baseline/
cargo run --release
```

## governor-bench: Criterion benchmark vs `governor` rate-limiter

`governor-bench/` compares `Budget::spend` against the popular
`governor` rate-limiter crate on construction time, memory
footprint, and per-spend latency.

Verified results from `cargo bench`:

| Operation | Budget | `governor` | Ratio |
|---|---:|---:|---:|
| Construction (`new`) | ~675 ps | ~234 ns | **~350× faster** |
| Memory footprint | 8 bytes | 72 bytes | **9× smaller** |
| Single spend (full receipt cycle) | ~1.21 µs | ~8.58 ns (`check`) | governor faster on this hot path |
| 100-spend sequence | ~1.24 µs total | ~860 ns total | governor faster |

The construction and memory comparisons demonstrate the
const-generic affine `Budget` is structurally simpler than
`governor::RateLimiter`. The per-spend hot path is faster on
`governor` because it doesn't do the receipt-cycle bookkeeping
that `Budget::spend_with_receipt` performs — `governor` is just an
atomic increment whereas `Budget::spend_with_receipt` allocates a
`Receipt` and threads the affine type-state.

```bash
cd governor-bench/
cargo bench
```

See paper §V-C6 for the appropriate "indicative pilot, not main-table
claim" framing.

## multiway: Python framework comparison

`multiway/` exercises the discipline across three Python LLM
frameworks: LangChain, LiteLLM, and AutoGen.

### Setup

```bash
cd multiway/
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# Set API keys
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

### Run

```bash
python -m harness --framework langchain --n 100
python -m harness --framework litellm --n 100
python -m harness --framework autogen --n 100
```

Output: per-framework CSV in `results/` showing cumulative spend,
refund rate, and any cap-soundness violations.

## Reproducing the paper's headline numbers

To regenerate the key empirical results, in order:

```bash
# 1. Contention sweeps + catalog (167 rows, 23 frameworks)
cd budget-spike/
./run_sweep.sh

# 2. Live API validation (5,424 row-events, 0 cap-soundness violations)
cd ../refund-live/
export ANTHROPIC_API_KEY="..." OPENAI_API_KEY="..." GEMINI_API_KEY="..."
cargo run --release --bin refund-live-1000

# 3. Governor benchmark
cd ../governor-bench/
cargo bench

# 4. Pre-call vs post-call comparison
cd ../fair-baseline/
cargo run --release

# 5. Cross-framework (Python)
cd ../multiway/
source .venv/bin/activate
python -m harness --all
```

## Honest scope on the corpus as a whole

What the empirical corpus **does** demonstrate:

- The receipt/refund cycle works correctly across providers, models,
  and workload types under the conditions tested.
- The discipline's construction is structurally cheaper than the
  comparable `governor` rate-limiter (memory, alloc).
- The catalog evidence (167 entries / 23 frameworks) is plausibly
  representative of the agent-spend failure space.
- TB achieves zero overshoot at all four tested cap settings; the
  callback baseline overshoots by 2-26%.

What it does **not** demonstrate:

- **Generalization to all providers/models/workloads.** The corpus
  covers Anthropic, OpenAI, Gemini, Groq, and self-hosted Ollama
  Llama-3.2 / DeepSeek-R1. Other providers (Cohere, AWS Bedrock,
  Mistral) are out of scope.
- **Long-term stability.** Sweeps were run over hours, not weeks.
- **Adversarial robustness.** A motivated attacker constructing
  pathological prompts could probably exceed the byte-length bound on
  some tokenizers; the corpus is not designed to find such cases.
- **A real-world incident reproduction.** We did not pay the $235 for
  the claude-code compaction loop ourselves; the incident's evidence
  is the cited GitHub issue, not our own runs.

These limitations are acknowledged in paper §V.

## Repository layout

```
token-budgets-experiments/
├── budget-spike/           # Contention sweeps + the catalog CSV (167 entries)
│   ├── src/
│   ├── benches/
│   ├── budget-archaeology.csv
│   └── Cargo.toml
├── refund-live/            # Live-API validation (10 binaries, 5,424 row-events)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── providers.rs
│   │   └── bin/
│   │       ├── a1-adversarial.rs
│   │       ├── a1-adversarial-n100.rs
│   │       ├── a1-rerun.rs
│   │       ├── compare-estimators.rs
│   │       ├── compare-estimators-openai.rs
│   │       ├── multi-turn-session.rs
│   │       ├── react-agent-bench.rs
│   │       ├── reasoning-eval.rs
│   │       ├── refund-live-1000.rs
│   │       └── refund-live-multi.rs
│   ├── sweep_results_*/    # Run outputs (CSV; 15 result files)
│   └── Cargo.toml
├── fair-baseline/          # TB vs Callback at 4 cap settings
│   ├── src/
│   └── Cargo.toml
├── governor-bench/         # Criterion vs governor crate
│   ├── benches/
│   └── Cargo.toml
├── multiway/               # Python: LangChain + LiteLLM + AutoGen
│   ├── harness/
│   ├── results/
│   └── pyproject.toml
├── .gitignore
├── README.md               # This file
├── LICENSE-MIT
└── LICENSE-APACHE
```

## Related repositories

| Repository | What it contains |
|---|---|
| [`token-budgets`](https://github.com/sajjadanwar0/token-budgets) | Main affine-API library + 167-entry catalog |
| [`token-budgets-extensions`](https://github.com/sajjadanwar0/token-budgets-extensions) | Adaptive estimator, Verus skeleton |
| [`token-budgets-formals`](https://github.com/sajjadanwar0/token-budgets-formals) | 5-tier mechanization (TLAPS / TLC / Coq / Dafny / Verus) |
| [`token-budgets-experiments`](https://github.com/sajjadanwar0/token-budgets-experiments) | This repo |
| [`rig-budget`](https://github.com/sajjadanwar0/rig-budget) | Integration with the `rig` LLM framework |

## Paper

```bibtex
@article{khan-token-budgets-2026,
  author  = {Khan, Sajjad},
  title   = {Token Budgets: An Affine-Resource Discipline for LLM Cost Caps in Rust},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```

Sections of the paper sourced from this artifact:
- §V: Empirical validation (refund-live + budget-spike)
- §V-C6: Governor comparison (governor-bench)
- §V (Table XXIV): Pre-call vs post-call (fair-baseline)
- §V: Multi-framework applicability (multiway)
- §V: Catalog (167 cases across 23 frameworks)

## License

Dual MIT/Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.