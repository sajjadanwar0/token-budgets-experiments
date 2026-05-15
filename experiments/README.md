# Token Budgets — Experiment Packages

Two runnable experiment packages addressing the two convergent reviewer concerns
from EMSE-target review rounds.

## Package 1: `anthropic_estimator/`

**Status**: Scripts ready. Requires `ANTHROPIC_API_KEY` and the
`token-budgets` Rust repo to run. Cost ~$0.50–$1.00 in API calls.

**Goal**: Re-run the 30 Anthropic tool-loop experiments with
`AnthropicEstimator` as the default (instead of byte-length). The byte-length
estimator failed A1 30/30; AnthropicEstimator (which uses Anthropic's actual
tokenizer) is expected to pass 30/30.

**Runtime**: ~30–45 min (API-rate-limited).

```bash
cd anthropic_estimator
export ANTHROPIC_API_KEY=...
export TB_ROOT=~/RustroverProjects/token-budgets  # adjust as needed
./run_experiment.sh
```

## Package 2: `conjecture_1_stress/`

**Status**: ✅ **EXECUTED AND VERIFIED in this container.** Results at
`conjecture_1_stress/RESULTS_VERIFIED_2026-05-15/`.

**Goal**: Adversarial stress-test of the Φ-invariant from Lemma 2 (safety
preservation under Tokio scheduling). 10,000 randomised concurrent
workloads under work-stealing.

**Results obtained**:
- 22,886,985 events processed by monitor
- ~320M total Budget operations
- **0 Φ-invariant violations**
- **0 conservation-invariant violations**
- Wall-clock: 17 seconds on this container's hardware

These are now in the paper as §V.J "Adversarial concurrent stress validation
of Lemma 2".

```bash
cd conjecture_1_stress
cargo build --release
./run_experiment.sh   # or: cargo run --release -- --iterations 10000
python3 analyze.py results
```

## Output Files Produced

After running, each experiment writes:
- `results/*.csv` – per-iteration raw data
- `results/summary.json` – aggregate summary
- `results/paper_update.md` – ready-to-paste paper-update text

## Build Requirements

- Rust: 1.93+ for the production token-budgets repo; 1.75+ for the stress test
- Python: 3.11+ with `anthropic` and `pandas` (for the Anthropic experiment)
