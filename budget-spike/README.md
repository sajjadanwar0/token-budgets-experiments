# Rust Artifact (§IV–§V of paper)

Rust implementation of the Token Budgets affine type, microbenchmarks,
real-API demos, and live boundary sweeps.

**Quick start:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
cargo test         # 12 tests
cargo bench        # ~30s, microbenchmarks
cargo run --bin demo_async_anthropic   # ~$0.001
```

## Directory map

```
.
├── Cargo.toml / Cargo.lock              Edition 2024, rustc 1.93+
├── src/
│   ├── lib.rs                           Affine Budget core, ~900 lines
│   ├── receipt.rs                       Reservation receipt (refund / forfeit)
│   ├── tokenizer.rs                     Conservative byte-length estimator
│   ├── llm_client.rs                    Trait + Mock + Anthropic + OpenAI
│   └── bin/                             5 demo binaries
│       ├── demo_runtime.rs              Sync runtime
│       ├── demo_async_mock.rs           Async, offline, free
│       ├── demo_async_anthropic.rs      Real Anthropic Haiku
│       ├── demo_async_openai.rs         Real OpenAI gpt-4o-mini
│       ├── demo_multi_provider.rs       Single Budget across providers
│       └── tc_live_harness.rs           Live boundary harness (runs sweeps)
│
├── tests/
│   ├── compile_fail/                    rustc-rejection tests (4)
│   │   ├── clone_attempt.rs             E0599 — Budget cannot be cloned
│   │   ├── double_spend.rs              E0382 — Budget cannot be double-spent
│   │   ├── spend_after_split.rs         E0382 — parent consumed by split
│   │   └── escape_via_return.rs         E0515 — lifetime-bound, no escape
│   └── async_integration/               tokio integration (4)
│       └── ...                          await + spawn boundaries
│
├── benches/                             Criterion microbenchmarks
│   ├── spend_overhead.rs                Pure-function benches
│   ├── end_to_end.rs                    Estimator path benches
│   └── prompts/                         Fixed prompts for E2E reproducibility
│
├── bench_results.txt                    spend() = 905 ps median
├── bench_results_e2e.txt                Estimator path bench
├── machine_info.txt                     Hardware/kernel/rustc reproducibility
├── FINDINGS.md                          Detailed empirical findings
├── smoke.csv                            Smoke test data
│
├── fill_anthropic_v_m.sh                Replay §V-M Anthropic batch
├── fill_groq_arg_halluc.sh              Replay §V-J Groq retry batch
├── sweep_boundary.sh                    Replay §V-B boundary sweep
│
└── sweep_results/
    ├── main/                            §V-B main: 9 CSVs × 10 rows = 90 runs
    │                                    (3 providers × 3 workloads × n=10)
    ├── boundary/                        §V-B boundary R1: cap-below-estimator
    ├── anthropic_cap2000_VM/            §V-M: 3 CSVs × 10 rows = 30 Anthropic runs
    │                                    + utilization_anthropic_2000.csv (the A1
    │                                    falsification finding)
    └── groq_retry/                      §V-J: 10 retry runs of arg-hallucination
                                         + utilization_groq_arg_halluc_retry.csv
```

## Empirical claims and where they live

| Claim | File |
|-------|------|
| `Budget::spend()` is sub-nanosecond | `bench_results.txt` (893 ps median) |
| Faster than `u64::checked_sub` | `bench_results.txt` (1366 ps for `checked_sub`) |
| Zero overhead vs raw subtract | `bench_results.txt` (-1.3% relative) |
| Compile-fail tests reject misuse | `tests/compile_fail/*.rs` + `*.stderr` |
| Affine semantics survive `.await` | `tests/async_integration::spend_then_await_then_spend` |
| 90 main sweep runs, zero overshoot | `sweep_results/main/*.csv` |
| 30 Anthropic cap=2000 runs (§V-M) | `sweep_results/anthropic_cap2000_VM/*.csv` |
| Provider-agnostic enforcement | `src/bin/demo_multi_provider.rs` |

## Limitations explicitly out of scope (per FINDINGS §5)

- Refund semantics on failed LLM calls (now partially in `receipt.rs`)
- Streaming responses (per-token deduction not implemented)
- Sub-budget hierarchies beyond depth 2
- Pricing accuracy (uses approximate per-token figures)
- Const-generic typestate (current Budget is runtime-checked u64)
- Formal soundness proof at type level (TLA+/Dafny mechanization in `../formals/`,
  Iris/RustBelt mechanization in progress in `../rustbelt-mechanization/`)

## Reproduction cost

Total cost of running every demo and the full sweep: **under $0.01 USD**.
