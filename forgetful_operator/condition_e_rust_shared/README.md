# Condition E — Rust shared Arc<Mutex<Budget>> with pre-flight reservation

**Status**: EXECUTED 2026-05-27, outcome (i) per paper §8.3 M7 pre-registration.
**Result**: 0/30 overshoot at $B_0 = 60$ uc on `claude-haiku-4-5`, $T=0$.
**Paper reference**: §5.11 Table 14 row E + §8.3 M7 EXECUTED status line.

This directory holds the M7 forgetful-operator Condition E harness:
a Rust implementation using `Arc<tokio::sync::Mutex<Budget>>` shared
across three `tokio::spawn`-ed children with the same pre-flight
reservation pattern as Condition B's Python locked variant. The
condition exists to isolate the integrity-layer contribution from the
allocation-strategy contribution in the A-vs-C contrast (see paper
§5.11 threats-to-validity).

## Contents

| File | Purpose |
|---|---|
| `Cargo.toml` | Rust dependencies (tokio, reqwest, csv, clap, anyhow, serde) |
| `src/main.rs` | The harness — single-file, ~440 LoC; pre-attack rationale in header doc comment |
| `condition_e_results.csv` | 90 rows (30 trials × 3 children); committed result from the 2026-05-27 run |
| `condition_e_run.log` | Console log from the canonical run; shows per-trial pattern + summary |
| `condition_e_analysis.txt` | `analyze_condition_e.py` output: 0/30 overshoot, Wilson 95% CI [0.000, 0.114], paper-ready Table 14 row |
| `analyze_condition_e.py` | Pure-stdlib aggregation script; encodes paper §5.11 baselines + pairwise Fisher exact |

## Reproducing the result

```bash
# Build (offline)
cargo build --release

# Verify the harness compiles without API key
cargo check --release

# Re-execute (requires Anthropic API key, ~$0.005 cost, ~5 min)
export ANTHROPIC_API_KEY=sk-ant-...
cargo run --release -- \
    --trials 30 --budget 60 --children 3 \
    --output condition_e_results_replay.csv \
    --model claude-haiku-4-5 \
    --temperature 0.0 --max-output-tokens 30 \
    --per-child-estimate 31

# Re-analyze
python3 analyze_condition_e.py \
    --results condition_e_results_replay.csv \
    --report condition_e_analysis_replay.txt

# Compare against committed result
diff condition_e_results.csv condition_e_results_replay.csv  
diff condition_e_analysis.txt condition_e_analysis_replay.txt
```

## Honest framing on the result

The 0/30 outcome confirms structural parity with Condition B (Python
locked) at these parameters: 3 children, shared $B_0=60$ uc, per-child
reservation 31 uc. Only one child fits pre-flight (31 ≤ 29 is false),
so children 1 and 2 refuse pre-flight; the refund of (31 − 1) = 30 uc
arrives after siblings have refused. The "1 admitted, 2 refused"
admit pattern matches Condition B exactly (paper Table 14: B has
admit/trial = 1.0/3, E has admit/trial = 1.0/3).

The mean spend differs (E: 1 uc, B: 23 uc) because the Condition E
prompt is smaller than the catalogue-derived prompt used for
Conditions A-D. This is disclosed in the Table 14 footnote in the
paper; the 31× over-reservation in Condition E is "load-bearing for
the 0/30 result and represents the worst-case-for-the-discipline
scenario."

The structural parity claim is supported on the admit/overshoot
dimensions; the lower mean spend reflects prompt size, not
discipline difference.
