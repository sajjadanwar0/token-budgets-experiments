use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use budget_spike::Budget;

const PROMPT_SHORT: &str = "What is 2+2?";
const PROMPT_MEDIUM: &str = include_str!("prompts/medium.txt");
const PROMPT_LONG: &str = include_str!("prompts/long.txt");

fn estimate_byte_length(prompt: &str) -> u64 {
    prompt.len() as u64
}

#[cfg(feature = "tiktoken")]
fn estimate_tiktoken(prompt: &str, encoder: &tiktoken_rs::CoreBPE) -> u64 {
    encoder.encode_ordinary(prompt).len() as u64
}

fn bench_byte_length_estimator(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_length_estimator");
    for (name, prompt) in &[("short", PROMPT_SHORT), ("medium", PROMPT_MEDIUM), ("long", PROMPT_LONG)] {
        group.bench_with_input(BenchmarkId::from_parameter(name), prompt, |b, &p| {
            b.iter(|| {
                let estimate = estimate_byte_length(black_box(p));
                black_box(estimate)
            });
        });
    }
    group.finish();
}

#[cfg(feature = "tiktoken")]
fn bench_tiktoken_o200k_estimator(c: &mut Criterion) {
    let encoder = tiktoken_rs::o200k_base().expect("o200k_base encoding");
    let mut group = c.benchmark_group("tiktoken_o200k_estimator");
    for (name, prompt) in &[("short", PROMPT_SHORT), ("medium", PROMPT_MEDIUM), ("long", PROMPT_LONG)] {
        group.bench_with_input(BenchmarkId::from_parameter(name), prompt, |b, &p| {
            b.iter(|| {
                let estimate = estimate_tiktoken(black_box(p), &encoder);
                black_box(estimate)
            });
        });
    }
    group.finish();
}

fn bench_end_to_end_byte_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_byte_length");
    for (name, prompt) in &[("short", PROMPT_SHORT), ("medium", PROMPT_MEDIUM), ("long", PROMPT_LONG)] {
        group.bench_with_input(BenchmarkId::from_parameter(name), prompt, |b, &p| {
            b.iter(|| {
                let budget = Budget::new(1_000_000_000);
                let estimate = estimate_byte_length(black_box(p));
                let (b2, _) = budget.spend(estimate, || ()).expect("budget large");
                black_box(b2);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "tiktoken")]
fn bench_end_to_end_tiktoken(c: &mut Criterion) {
    let encoder = tiktoken_rs::o200k_base().expect("o200k_base encoding");
    let mut group = c.benchmark_group("end_to_end_tiktoken");
    for (name, prompt) in &[("short", PROMPT_SHORT), ("medium", PROMPT_MEDIUM), ("long", PROMPT_LONG)] {
        group.bench_with_input(BenchmarkId::from_parameter(name), prompt, |b, &p| {
            b.iter(|| {
                let budget = Budget::new(1_000_000_000);
                let estimate = estimate_tiktoken(black_box(p), &encoder);
                let (b2, _) = budget.spend(estimate, || ()).expect("budget large");
                black_box(b2);
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "tiktoken"))]
criterion_group!(
    benches,
    bench_byte_length_estimator,
    bench_end_to_end_byte_length
);

#[cfg(feature = "tiktoken")]
criterion_group!(
    benches,
    bench_byte_length_estimator,
    bench_tiktoken_o200k_estimator,
    bench_end_to_end_byte_length,
    bench_end_to_end_tiktoken
);

criterion_main!(benches);
