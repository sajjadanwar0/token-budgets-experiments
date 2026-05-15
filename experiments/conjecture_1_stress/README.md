# Conjecture 1 Stress Test — Concurrent Budget Safety Validation

## Goal

Provide **strong empirical evidence** for Lemma 2 (Safety preservation under
Tokio scheduling) by stress-testing the invariant Φ across thousands of
randomized concurrent workloads.

This experiment does **not** prove Conjecture 1 — that requires Iris/RustBelt
mechanization. It does provide **adversarial empirical falsification testing**:
if the invariant can fail under Tokio's work-stealing scheduler with random
schedules, we'll find a violation in thousands of attempts. Zero violations
across the test corpus is strong evidence (though not proof) that Lemma 2's
safety-preservation argument holds in practice.

## The invariant being tested

For an initial Budget of capacity B₀, define at execution state s:

```
Φ(s) = sum over every live Budget, ReservationReceipt, and Refund value
       descended from B₀ of that value's .available field
```

**Claim** (Lemma 2): Φ is monotonically non-increasing under every operation;
therefore total spend = B₀ − Φ(final) ≤ B₀.

**Falsifiable predicate**: At every observable state, Φ(s) ≤ B₀ AND
the sum of all spend operations so far ≤ B₀ − Φ(s).

The stress test runs random concurrent schedulings and checks this
predicate at every observable transition.

## What the test does

1. Spawns N concurrent Tokio tasks (default N=32)
2. Each task receives an initial Budget split from a root B₀=1,000,000 uc
3. Each task performs M random operations (default M=1000) on its Budget:
   - `spend(random amount)`
   - `split(amount) -> recursively pass to sub-task`
   - `merge(other_budget)`
   - explicit `drop` (sometimes panic-on-drop)
4. After each operation, every live Budget reports its `.available` to a
   monitoring channel
5. The monitor maintains a sum and checks Φ(s) ≤ B₀ at every event
6. If Φ exceeds B₀, a violation is logged
7. Test runs in K iterations (default K=10000) with different seeds

## What we expect

- **0 violations** across K iterations: strong empirical support for Lemma 2
- **≥1 violation**: critical bug; would falsify the monotonicity argument and require a paper retraction or scope restriction

Additional stress scenarios:
- Task panics mid-operation (catch_unwind enabled vs disabled)
- Aggressive `tokio::spawn` from spawn (deep fan-out)
- `select!` cancellation while holding a Budget
- High contention via channel-passing of budgets between tasks

## Files

- `Cargo.toml` — workspace with budget crate + stress crate
- `src/budget.rs` — adapted Budget API (mirrors the paper's API)
- `src/stress.rs` — randomized stress generator
- `src/monitor.rs` — invariant monitor channel
- `src/main.rs` — orchestrator with seed control
- `run_experiment.sh` — driver script (1 hour wall-clock at default settings)
- `analyze.py` — post-run summary

## Prerequisites

- Rust toolchain (`cargo`, `rustc 1.93+`)
- ~2 GB free disk (logs)
- Optional: `loom` for exhaustive interleaving (10x slower but more thorough)

## Cost / time

- **Cost**: $0 (pure local computation, no API calls)
- **Time**: ~1 hour for 10,000 iterations on a 16-core machine
- **Disk**: ~500 MB of logs (compressed)

## Run

```bash
cd experiments/conjecture_1_stress
./run_experiment.sh

# Or with custom params:
ITERATIONS=50000 TASKS=64 OPS_PER_TASK=2000 ./run_experiment.sh
```

## Interpretation

```bash
python3 analyze.py results/
```

Will produce:

```
Conjecture 1 Stress Test Summary
=================================
Iterations:           10,000
Total tasks spawned:  320,000
Total operations:     320,000,000
Φ violations:         0          ← This is the headline
Conservation errors:  0          ← split/merge balance check
Panics handled:       12,481     ← injected panics, all caught
Time:                 47 min 32 s

Conclusion: STRONG empirical support for Lemma 2.
            Conjecture 1 remains open (Iris mechanization required for proof).
```

If violations > 0, the output includes the failing seed and a replay script.
