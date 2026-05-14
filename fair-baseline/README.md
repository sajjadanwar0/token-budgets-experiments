# fair-baseline: Token Budgets vs Runtime Callback (Fair Comparison)

Replay an Anthropic workload through two enforcement mechanisms with
**identical cost arithmetic**, differing only in **when** the cap check
fires. The point: any overshoot the post-call callback shows is a
structural property of post-call enforcement, not an arithmetic defect
or "designed for the wrong task" artifact.

## Setup

```
fair-baseline/
├── Cargo.toml
├── README.md
└── src/
    └── main.rs
```

Both Cargo.toml has an empty `[workspace]` table so it stands alone as
its own workspace root (doesn't get pulled into a parent workspace).

## Usage

```bash
cd fair-baseline
# Point at the a1-rerun CSV produced by the a1-rerun binary:
cargo run --release -- --input ../a1-rerun/a1_rerun_results.csv

# Or supply a path to wherever you have the CSV:
cargo run --release -- --input /path/to/a1_rerun_results.csv
```

The cap sweep is configurable via repeated `--cap-uc` flags; default is
5000, 6500, 8000, 10000.

## What it reproduces

Table XVI in the paper (§V-M, Fair-baseline replay). Key numbers at
cap = 6500 uc:

| Mechanism | Admitted | Spent (uc) | Overshoot (uc) |
|-----------|----------|------------|----------------|
| TB (pre-call) | 2/30 | 4,086 | **0** |
| Callback (post-call) | 4/30 | 8,174 | 1,674 (25.8% of cap) |

No API calls are made; the workload data is the captured a1-rerun cell.
Runtime is sub-second.

## What this isolates

The runtime mitigations the paper compares against in §V-D were not
designed to bound dollars. This binary builds a runtime mitigation that
*is* designed to bound dollars (same estimator, same arithmetic, same
cap) and shows the gap that remains:

- TB: 0 overshoot, fewer calls admitted (conservative estimator)
- Callback: ~one call's actual cost of overshoot, more calls admitted

The gap is the structural cost of observing-then-acting versus
deciding-before-acting.
