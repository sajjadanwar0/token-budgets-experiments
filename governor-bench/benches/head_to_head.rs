//! Head-to-head benchmark: Token Budgets affine discipline vs.
//! `governor` rate-limiter. Same workload, both libraries, measure
//! per-operation latency, throughput, and memory footprint.

use token_budgets::Budget;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use std::sync::Arc;
use std::time::Duration;

const CAP: u64 = 1_000_000;
type B = Budget<CAP>;

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    group.bench_function("Budget::new", |b| {
        b.iter(|| black_box(B::new(black_box(500_000)).unwrap()))
    });

    group.bench_function("governor::RateLimiter::direct", |b| {
        b.iter(|| {
            black_box(RateLimiter::direct(Quota::per_second(nonzero!(500_000u32))))
        })
    });

    group.finish();
}

fn bench_single_spend(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_spend");
    group.throughput(Throughput::Elements(1));

    group.bench_function("Budget::spend", |b| {
        b.iter_with_setup(
            || B::new(1_000_000).unwrap(),
            |budget| {
                let _ = black_box(budget.spend(black_box(100))).unwrap();
            },
        )
    });

    group.bench_function("governor::check", |b| {
        let limiter = Arc::new(RateLimiter::direct(
            Quota::per_second(nonzero!(1_000_000u32)).allow_burst(nonzero!(1_000_000u32))
        ));
        b.iter(|| {
            black_box(limiter.check().is_ok())
        })
    });

    group.bench_function("governor::check_n(100)", |b| {
        let limiter = Arc::new(RateLimiter::direct(
            Quota::per_second(nonzero!(1_000_000u32)).allow_burst(nonzero!(1_000_000u32))
        ));
        b.iter(|| {
            black_box(limiter.check_n(nonzero!(100u32)).is_ok())
        })
    });

    group.finish();
}

fn bench_sequence_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_100_spends");
    group.throughput(Throughput::Elements(100));

    group.bench_function("Budget::spend x100", |b| {
        b.iter_with_setup(
            || B::new(1_000_000).unwrap(),
            |mut budget| {
                for _ in 0..100 {
                    budget = black_box(budget.spend(black_box(100)).unwrap());
                }
                black_box(budget);
            },
        )
    });

    group.bench_function("governor::check x100", |b| {
        let limiter = Arc::new(RateLimiter::direct(
            Quota::per_second(nonzero!(1_000_000u32)).allow_burst(nonzero!(1_000_000u32))
        ));
        b.iter(|| {
            for _ in 0..100 {
                black_box(limiter.check().is_ok());
            }
        })
    });

    group.finish();
}

fn bench_memory_footprint(c: &mut Criterion) {
    // Measure size_of_val for both types
    let budget = B::new(1_000_000).unwrap();
    let limiter = RateLimiter::direct(Quota::per_second(nonzero!(1_000_000u32)));

    println!("\n=== Memory footprint ===");
    println!("Budget<1_000_000>:     {} bytes", std::mem::size_of_val(&budget));
    println!("governor::RateLimiter: {} bytes", std::mem::size_of_val(&limiter));
    // Note: governor stores GCRA state and clock; Budget stores only u64.

    let _ = c.bench_function("noop", |b| b.iter(|| 1));
}

criterion_group!(
    benches,
    bench_construction,
    bench_single_spend,
    bench_sequence_100,
    bench_memory_footprint
);
criterion_main!(benches);
