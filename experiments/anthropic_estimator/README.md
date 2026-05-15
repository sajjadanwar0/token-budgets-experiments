# AnthropicEstimator A1 Validation Experiment

## Goal

Verify that the `AnthropicEstimator` (using Anthropic's actual tokenizer via `tiktoken-rs` or equivalent) satisfies Assumption A1 on the same tool-loop workloads that caused the 30/30 byte-length failures.

**A1 statement**: For every prompt `p` sent to Anthropic, the estimator's output upper-bounds the provider's billed input tokens:

```
E(p) >= t_in(p)
```

The byte-length estimator failed this on 30/30 runs because Anthropic's tool-call encoding uses short special tokens that the byte-count underestimates. The `AnthropicEstimator` runs Anthropic's actual tokenizer on the prompt, so it should match (or near-match) the billed token count.

## What we expect

| Estimator | A1 holds | B/T ratio expected |
|---|---|---|
| ByteLength (v1-v9) | 0/30 | 0.72-0.79 (fails) |
| AnthropicEstimator (this experiment) | 30/30 (target) | 1.00 ± epsilon (tight) |

The "epsilon" reflects whatever residual server-side rewriting Anthropic does (system-prompt injection, etc). If `AnthropicEstimator` holds 30/30, the headline becomes "stratified default satisfies A1 across all 3 providers." If it fails any runs, those failures characterize the server-side-rewriting gap.

## Files

- `run_experiment.sh` — orchestrator (depends on your existing `tc_live_harness` binary)
- `rust_patch.diff` — patch for `token-budgets/src/estimator/default.rs` to flip Anthropic to AnthropicEstimator
- `runner.py` — Python harness for the 30 runs (3 workloads × 10 runs each)
- `analyze.py` — compute B/T ratio per run, check A1 hold rate, print summary
- `expected_output.md` — what the final analysis output should look like

## Prerequisites

- `ANTHROPIC_API_KEY` environment variable set
- `token-budgets` repo cloned at `$TB_ROOT` (default `~/RustroverProjects/token-budgets`)
- `token-budgets-experiments` repo cloned at `$TBE_ROOT` (default `~/RustroverProjects/token-budgets-experiments`)
- Rust toolchain (`cargo`, `rustc 1.93+`)
- Python 3.11+ with `anthropic` and `pandas` packages

## Cost estimate

30 runs × ~10 calls each × ~2000 tokens each × Anthropic Haiku pricing
= 30 × 10 × 2000 × $0.80/1M input + comparable output
≈ **$0.50–$1.00 total**

## Time estimate

- Apply patch + rebuild: 5 min
- Run 30 experiments: 30–45 min wall-clock (with API rate limits)
- Analyze results: 1 min
- **Total: ~1 hour**

## Run order

```bash
# 1. Apply the Rust patch
cd $TB_ROOT
git apply $EXP_DIR/rust_patch.diff
cargo build --release --features anthropic-estimator

# 2. Run the experiment
cd $EXP_DIR
export ANTHROPIC_API_KEY=...
./run_experiment.sh

# 3. Analyze
python3 analyze.py results/runs.csv
```

Output should be at `results/a1_validation.json` and `results/summary.md`.
