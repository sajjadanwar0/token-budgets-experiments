//! Microbenchmark: per-operation overhead of the affine Budget API.
//!
//! Run with: cargo bench
//!
//! Compares Budget operations against unguarded u64 baselines to measure
//! the compile-time-enforcement tax. Expected: single-digit nanoseconds
//! for spend(), comparable to bounds-checking on array indexing.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use budget_spike::{Budget, BudgetError, estimate_cost};

// =============================================================================
// Group 1: Construction
// =============================================================================

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    group.bench_function("budget_new", |b| {
        b.iter(|| {
            let budget = Budget::new(black_box(1_000_000));
            black_box(budget)
        })
    });

    // Baseline: just constructing a u64
    group.bench_function("u64_baseline", |b| {
        b.iter(|| {
            let n: u64 = black_box(1_000_000);
            black_box(n)
        })
    });

    group.finish();
}

// =============================================================================
// Group 2: Spend - success path
// =============================================================================

fn bench_spend_success(c: &mut Criterion) {
    let mut group = c.benchmark_group("spend_success");

    // Headline: Budget::spend on the success path
    group.bench_function("budget_spend_ok", |b| {
        b.iter(|| {
            let budget = Budget::new(black_box(1_000_000));
            let (remaining, _) = budget
                .spend(black_box(100), || ())
                .expect("must succeed");
            black_box(remaining)
        })
    });

    // Baseline: unguarded u64 subtraction
    group.bench_function("u64_subtract", |b| {
        b.iter(|| {
            let mut n: u64 = black_box(1_000_000);
            n -= black_box(100);
            black_box(n)
        })
    });

    // Baseline: u64 with bounds check (simulating what spend does internally)
    group.bench_function("u64_checked_sub", |b| {
        b.iter(|| {
            let n: u64 = black_box(1_000_000);
            let amount: u64 = black_box(100);
            let result = if amount > n {
                None
            } else {
                Some(n - amount)
            };
            black_box(result)
        })
    });

    group.finish();
}

// =============================================================================
// Group 3: Spend - insufficient path
// =============================================================================

fn bench_spend_insufficient(c: &mut Criterion) {
    let mut group = c.benchmark_group("spend_insufficient");

    group.bench_function("budget_spend_err", |b| {
        b.iter(|| {
            let budget = Budget::new(black_box(50));
            let result = budget.spend(black_box(100), || ());
            // Should be Err - we want to bench the error path
            assert!(matches!(result, Err(BudgetError::Insufficient { .. })));
            black_box(result)
        })
    });

    group.finish();
}

// =============================================================================
// Group 4: Split + merge round-trip (multi-agent pattern)
// =============================================================================

fn bench_split_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_merge");

    group.bench_function("split_then_merge", |b| {
        b.iter(|| {
            let parent = Budget::new(black_box(1_000_000));
            let (parent, child) = parent
                .split(black_box(400_000))
                .expect("split must succeed");
            let merged = parent.merge(child);
            black_box(merged)
        })
    });

    group.finish();
}

// =============================================================================
// Group 5: estimate_cost helper
// =============================================================================

fn bench_estimate_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_cost");

    group.bench_function("estimate_cost_call", |b| {
        b.iter(|| {
            let cost = estimate_cost(
                black_box(1000),
                black_box(500),
                black_box(3_000_000),
            );
            black_box(cost)
        })
    });

    group.finish();
}

// =============================================================================

criterion_group!(
    benches,
    bench_construction,
    bench_spend_success,
    bench_spend_insufficient,
    bench_split_merge,
    bench_estimate_cost,
);
criterion_main!(benches);