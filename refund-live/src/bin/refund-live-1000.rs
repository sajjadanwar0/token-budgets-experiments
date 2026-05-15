//! 1000-call live API regime with statistical analysis of the A1
//! margin distribution and conservation verification across the
//! full run. Companion to the 10-call refund-live smoke test.
//!
//! Usage: N_CALLS=1000 ANTHROPIC_API_KEY=sk-ant-... \
//!          cargo run --release --bin refund-live-1000
//!
//! Estimated cost: ~$0.10-3 depending on N_CALLS and prompt length.

use anyhow::{Context, Result};
use token_budgets::Budget;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const ANTHROPIC_PER_IN_TOKEN_UC: u64 = 1;
const ANTHROPIC_PER_OUT_TOKEN_UC: u64 = 5;
const BUDGET_CAP: u64 = 100_000_000; // $100
const MAX_OUT_TOKENS: u32 = 200;
const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const MODEL: &str = "claude-haiku-4-5-20251001";

type B = Budget<BUDGET_CAP>;

#[derive(Debug, Serialize)]
struct AnthropicReq<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<Msg<'a>>,
}

#[derive(Debug, Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicResp {
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

fn build_prompt(i: usize) -> String {
    let class = i % 10;
    match class {
        0 => format!("What is the capital of country #{}?", i % 200),
        1 => format!("Briefly: explain the concept '{}'.", topic(i)),
        2 => format!("Define: {}", topic(i)),
        3 => format!("List three examples of {}.", topic(i)),
        4 => format!("What year was {} invented?", topic(i)),
        5 => format!("Is {} larger than {}?", topic(i), topic(i + 1)),
        6 => format!("Name one scientist associated with {}.", topic(i)),
        7 => format!("Briefly compare {} and {}.", topic(i), topic(i + 1)),
        8 => format!("Output JSON: {{\"topic\": \"{}\", \"score\": ?}}", topic(i)),
        _ => format!("Suggest one fact about {}.", topic(i)),
    }
}

fn topic(i: usize) -> &'static str {
    const TOPICS: &[&str] = &[
        "photosynthesis", "machine learning", "tensor", "graphene", "DNA",
        "Mars", "asteroid", "supernova", "string theory", "TCP/IP",
        "Bayesian inference", "kernel methods", "regularisation", "transformer",
        "attention", "self-attention", "RNN", "gradient descent", "Adam optimiser",
        "dropout", "batch normalisation", "encoder", "decoder", "perceptron",
        "ImageNet", "BERT", "GPT", "ResNet", "VGG", "AlexNet",
    ];
    TOPICS[i % TOPICS.len()]
}

#[derive(Debug, Serialize)]
struct CallRecord {
    idx: usize,
    reservation_uc: u64,
    actual_uc: u64,
    refund_uc: u64,
    input_tokens: u64,
    output_tokens: u64,
    latency_ms: u128,
    margin_ratio: f64,
}

fn compute_reservation(body_bytes: usize) -> u64 {
    body_bytes as u64 * ANTHROPIC_PER_IN_TOKEN_UC
        + (MAX_OUT_TOKENS as u64) * ANTHROPIC_PER_OUT_TOKEN_UC
}

#[tokio::main]
async fn main() -> Result<()> {
    let n_calls: usize = env::var("N_CALLS")
        .unwrap_or_else(|_| "1000".to_string())
        .parse()
        .context("N_CALLS must be a number")?;
    let api_key = env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY must be set")?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    let mut budget: Option<B> = Some(Budget::new(BUDGET_CAP)?);
    let mut records: Vec<CallRecord> = Vec::with_capacity(n_calls);
    let mut sum_reserved: u128 = 0;
    let mut sum_actual: u128 = 0;
    let mut sum_refund: u128 = 0;
    let mut a1_violations = 0usize;
    let mut api_errors = 0usize;

    println!(
        "=== Running {} calls against {}, cap = ${:.2} ===",
        n_calls, MODEL, BUDGET_CAP as f64 / 1e6
    );
    println!(
        "{:>4} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6} {:>6}",
        "#", "reserve", "actual", "refund", "in_tok", "out_tok", "ms", "x"
    );

    let run_start = Instant::now();

    for i in 0..n_calls {
        let prompt = build_prompt(i);
        let req = AnthropicReq {
            model: MODEL,
            max_tokens: MAX_OUT_TOKENS,
            messages: vec![Msg { role: "user", content: &prompt }],
        };
        let body = serde_json::to_string(&req)?;
        let reservation = compute_reservation(body.len());

        let current = budget.take().expect("budget present");
        let (after_reserve, receipt) = match current.spend_with_receipt(reservation) {
            Ok(x) => x,
            Err(e) => {
                println!("Budget exhausted at call {}: {:?}", i, e);
                budget = Some(B::new(0).unwrap_or_else(|_| {
                    panic!("cannot recover budget after exhaustion")
                }));
                break;
            }
        };

        let start = Instant::now();
        let resp = client
            .post(ANTHROPIC_URL)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await;
        let latency_ms = start.elapsed().as_millis();

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                api_errors += 1;
                eprintln!("API error at call {}: {}", i, e);
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
        };
        if !resp.status().is_success() {
            api_errors += 1;
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            eprintln!("API returned {} at call {}: {}", status, i, text);
            receipt.forfeit();
            budget = Some(after_reserve);
            continue;
        }
        let parsed: AnthropicResp = resp.json().await?;
        let actual = parsed.usage.input_tokens * ANTHROPIC_PER_IN_TOKEN_UC
            + parsed.usage.output_tokens * ANTHROPIC_PER_OUT_TOKEN_UC;

        if actual > reservation {
            a1_violations += 1;
            println!(
                "⚠ A1 violation at call {}: actual {} > reserved {}",
                i, actual, reservation
            );
            receipt.forfeit();
            budget = Some(after_reserve);
            continue;
        }

        let refund = receipt.confirm(actual)?;
        let refund_amount = refund.amount();
        budget = Some(refund.apply_to(after_reserve)?);

        sum_reserved += reservation as u128;
        sum_actual += actual as u128;
        sum_refund += refund_amount as u128;

        let ratio = reservation as f64 / actual.max(1) as f64;
        records.push(CallRecord {
            idx: i,
            reservation_uc: reservation,
            actual_uc: actual,
            refund_uc: refund_amount,
            input_tokens: parsed.usage.input_tokens,
            output_tokens: parsed.usage.output_tokens,
            latency_ms,
            margin_ratio: ratio,
        });

        if i % 50 == 0 || i == n_calls - 1 {
            println!(
                "{:>4} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6} {:>6.2}",
                i, reservation, actual, refund_amount,
                parsed.usage.input_tokens, parsed.usage.output_tokens,
                latency_ms, ratio
            );
        }
    }

    let run_elapsed = run_start.elapsed();
    let final_budget = budget.unwrap();
    let final_value = final_budget.micro_cents();

    // Statistical analysis
    let ratios: Vec<f64> = records.iter().map(|r| r.margin_ratio).collect();
    let mean = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;
    let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / ratios.len().max(1) as f64;
    let stddev = var.sqrt();
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if !sorted.is_empty() { sorted[sorted.len() / 2] } else { 0.0 };
    let p95 = if !sorted.is_empty() { sorted[sorted.len() * 95 / 100] } else { 0.0 };
    let p99 = if !sorted.is_empty() { sorted[sorted.len() * 99 / 100] } else { 0.0 };
    let min = if !sorted.is_empty() { sorted[0] } else { 0.0 };
    let max = if !sorted.is_empty() { sorted[sorted.len() - 1] } else { 0.0 };

    println!();
    println!("=== {}-call summary ===", n_calls);
    println!("Run time:         {:.2}s", run_elapsed.as_secs_f64());
    println!("Successful:       {}", records.len());
    println!("A1 violations:    {} ({:.2}%)",
             a1_violations, 100.0 * a1_violations as f64 / n_calls as f64);
    println!("API errors:       {}", api_errors);
    println!();
    println!("Initial budget:   ${:.6}", BUDGET_CAP as f64 / 1e6);
    println!("Total reserved:   ${:.6}", sum_reserved as f64 / 1e6);
    println!("Total actual:     ${:.6}", sum_actual as f64 / 1e6);
    println!("Total refunded:   ${:.6}", sum_refund as f64 / 1e6);
    println!("Final budget:     ${:.6}", final_value as f64 / 1e6);
    println!();
    println!("=== A1 margin distribution (reservation / actual) ===");
    println!("min:    {:.3}x", min);
    println!("p50:    {:.3}x", median);
    println!("mean:   {:.3}x (stddev {:.3})", mean, stddev);
    println!("p95:    {:.3}x", p95);
    println!("p99:    {:.3}x", p99);
    println!("max:    {:.3}x", max);
    println!();

    // Conservation
    let expected_final = BUDGET_CAP - sum_actual as u64;
    if final_value == expected_final {
        println!("✓ Conservation: initial - actual = final (exact, {} uc)", final_value);
    } else {
        println!(
            "✗ Conservation broken: expected {} got {} (diff {})",
            expected_final, final_value,
            (final_value as i128) - (expected_final as i128)
        );
    }

    // CSV
    let mut csv = File::create("refund_live_1000_results.csv")?;
    writeln!(csv, "idx,reservation_uc,actual_uc,refund_uc,input_tokens,output_tokens,latency_ms,margin_ratio")?;
    for r in &records {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{:.6}",
            r.idx, r.reservation_uc, r.actual_uc, r.refund_uc,
            r.input_tokens, r.output_tokens, r.latency_ms, r.margin_ratio
        )?;
    }
    println!();
    println!("Wrote {} rows to refund_live_1000_results.csv", records.len());

    Ok(())
}
