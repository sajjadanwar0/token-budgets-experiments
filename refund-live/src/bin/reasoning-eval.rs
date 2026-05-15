//! Live evaluation against OpenAI reasoning models (o-series).
//!
//! Reasoning tokens are included in `completion_tokens` and billed as
//! output. The discipline works unmodified: reserve max_completion_tokens
//! * out_rate, confirm with actual completion_tokens (which includes
//! reasoning tokens). Tests whether reasoning-model output variability
//! breaks A1 in practice.
//!
//! Usage: OPENAI_API_KEY=... cargo run --release --bin reasoning-eval
//! Cost: ~$5-10 on o3-mini for N=100 (depends on reasoning depth).

use anyhow::{Context, Result};
use token_budgets::Budget;
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const BUDGET_CAP: u64 = 100_000_000_000;  // $100 in nano-cents
type B = Budget<BUDGET_CAP>;

// o3-mini pricing: $1.10/M input, $4.40/M output
const IN_RATE_NC: u64 = 1100;
const OUT_RATE_NC: u64 = 4400;

struct ReasoningRecord {
    idx: usize,
    reservation_nc: u64,
    actual_nc: u64,
    refund_nc: u64,
    input_tokens: u64,
    completion_tokens: u64,
    reasoning_tokens: u64,
    visible_output_tokens: u64,
    latency_ms: u128,
    margin_ratio: f64,
    output_capped: bool,
}

fn build_prompt(i: usize) -> String {
    // Reasoning-favoring prompts: math, logic, multi-step problems
    match i % 10 {
        0 => format!("If 3x + 7 = 22, what is x? Show your reasoning step by step."),
        1 => format!("A train leaves city A at 60mph. Another leaves city B at 80mph going the other way. Cities are 350 miles apart. When do they meet? Reason it out."),
        2 => format!("Is the number {} prime? Explain why or why not.", 1000 + i),
        3 => format!("What's the next number in: 2, 6, 12, 20, 30, ?  Show reasoning."),
        4 => format!("Solve: x^2 - 5x + 6 = 0. Show all steps."),
        5 => format!("If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly? Explain."),
        6 => format!("Three switches control three light bulbs in another room. You can only enter the room once. How do you determine which switch controls which bulb?"),
        7 => format!("What's the probability of rolling at least one 6 in 4 dice rolls? Show working."),
        8 => format!("If today is Wednesday, what day of the week was it 100 days ago? Explain."),
        _ => format!("Compare and contrast two sorting algorithms in 100 words: bubble sort and merge sort. Be specific about complexity."),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY must be set")?;
    let model = env::var("REASONING_MODEL").unwrap_or_else(|_| "o3-mini".to_string());
    let max_completion_tokens: u32 = env::var("MAX_COMPLETION_TOKENS")
        .unwrap_or_else(|_| "4096".to_string())
        .parse()?;
    let n_calls: usize = env::var("N_CALLS")
        .unwrap_or_else(|_| "100".to_string())
        .parse()?;
    let reasoning_effort = env::var("REASONING_EFFORT")
        .unwrap_or_else(|_| "medium".to_string()); // low | medium | high

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300)) // reasoning can be slow
        .build()?;

    println!("=== Reasoning model live eval: {} ===", model);
    println!("max_completion_tokens: {}", max_completion_tokens);
    println!("reasoning_effort:      {}", reasoning_effort);
    println!("N: {}, budget cap: $100", n_calls);

    let mut budget: Option<B> = Some(Budget::new(BUDGET_CAP)?);
    let mut records: Vec<ReasoningRecord> = Vec::with_capacity(n_calls);
    let mut sum_r = 0u128;
    let mut sum_a = 0u128;
    let mut sum_reasoning = 0u128;
    let mut violations = 0usize;
    let mut api_errors = 0usize;

    let run_start = Instant::now();

    for i in 0..n_calls {
        let prompt = build_prompt(i);
        let req = json!({
            "model": &model,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": &reasoning_effort,
            "messages": [{"role": "user", "content": prompt}],
        });
        let body = serde_json::to_string(&req)?;
        let body_bytes = body.len() as u64;
        let reservation = body_bytes * IN_RATE_NC
            + (max_completion_tokens as u64) * OUT_RATE_NC;

        let current = budget.take().expect("budget present");
        let (after_reserve, receipt) = match current.spend_with_receipt(reservation) {
            Ok(x) => x,
            Err(e) => {
                println!("Budget exhausted at call {}: {:?}", i, e);
                break;
            }
        };

        let start = Instant::now();
        let resp = client.post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .body(body)
            .send().await;
        let latency_ms = start.elapsed().as_millis();

        let resp = match resp {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                api_errors += 1;
                let s = r.status();
                let t = r.text().await.unwrap_or_default();
                eprintln!("call {} HTTP {}: {}", i, s, &t[..t.len().min(200)]);
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
            Err(e) => {
                api_errors += 1;
                eprintln!("call {} err: {}", i, e);
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
        };

        let parsed: Value = resp.json().await?;
        let usage = match parsed.get("usage") {
            Some(u) => u,
            None => {
                api_errors += 1;
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
        };

        let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
        let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
        let reasoning_tokens = usage
            .get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let visible_output = completion_tokens.saturating_sub(reasoning_tokens);

        // Actual = input * in_rate + completion * out_rate
        // (completion includes both visible output AND reasoning tokens)
        let actual = input_tokens * IN_RATE_NC
            + completion_tokens * OUT_RATE_NC;

        if actual > reservation {
            violations += 1;
            println!("⚠ A1 VIOLATION call {}: reserve={} actual={} in={} comp={} (reason={}, vis={})",
                     i, reservation, actual, input_tokens, completion_tokens,
                     reasoning_tokens, visible_output);
            receipt.forfeit();
            budget = Some(after_reserve);
            continue;
        }

        let refund = receipt.confirm(actual)?;
        let refund_amount = refund.amount();
        budget = Some(refund.apply_to(after_reserve)?);

        sum_r += reservation as u128;
        sum_a += actual as u128;
        sum_reasoning += (reasoning_tokens * OUT_RATE_NC) as u128;

        records.push(ReasoningRecord {
            idx: i, reservation_nc: reservation, actual_nc: actual,
            refund_nc: refund_amount, input_tokens, completion_tokens,
            reasoning_tokens, visible_output_tokens: visible_output,
            latency_ms,
            margin_ratio: reservation as f64 / actual.max(1) as f64,
            output_capped: (completion_tokens as u32) == max_completion_tokens,
        });

        if i % 10 == 0 || i == n_calls - 1 {
            println!("{:>3} reserve={:>10} actual={:>10} in={:>4} comp={:>5} (reason={:>4}, vis={:>4}) ms={:>5} x={:>5.2}",
                     i, reservation, actual, input_tokens, completion_tokens,
                     reasoning_tokens, visible_output, latency_ms,
                     reservation as f64 / actual.max(1) as f64);
        }
    }

    let elapsed = run_start.elapsed();
    let mut ratios: Vec<f64> = records.iter().map(|r| r.margin_ratio).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ratios.len();
    let pct = |p: f64| if n == 0 { 0.0 } else { ratios[((n as f64 * p) as usize).min(n - 1)] };

    let reasoning_total: u64 = records.iter().map(|r| r.reasoning_tokens).sum();
    let visible_total: u64 = records.iter().map(|r| r.visible_output_tokens).sum();
    let reasoning_fraction = reasoning_total as f64 / (reasoning_total + visible_total).max(1) as f64;

    println!();
    println!("=== Reasoning model summary: {} | mct={} | effort={} ===",
             model, max_completion_tokens, reasoning_effort);
    println!("Wall time:        {:.1} min", elapsed.as_secs_f64() / 60.0);
    println!("Successful:       {}", records.len());
    println!("A1 violations:    {}", violations);
    println!("API errors:       {}", api_errors);
    println!();
    println!("Actual cost:      ${:.4}", sum_a as f64 / 1e9);
    println!("Reasoning cost:   ${:.4} ({:.1}% of output spend)",
             sum_reasoning as f64 / 1e9, 100.0 * reasoning_fraction);
    println!("Over-reservation: {:.3}x", sum_r as f64 / sum_a.max(1) as f64);
    println!();
    println!("Margin: min={:.3}x p50={:.3}x mean={:.3}x p95={:.3}x max={:.3}x",
             pct(0.0), pct(0.5),
             ratios.iter().sum::<f64>() / n.max(1) as f64,
             pct(0.95), pct(1.0));
    println!();
    println!("Reasoning fraction by call:");
    println!("  reasoning/completion ratio min={:.2}% mean={:.2}% max={:.2}%",
             100.0 * records.iter().map(|r| r.reasoning_tokens as f64 / r.completion_tokens.max(1) as f64).fold(f64::INFINITY, f64::min),
             100.0 * records.iter().map(|r| r.reasoning_tokens as f64 / r.completion_tokens.max(1) as f64).sum::<f64>() / records.len().max(1) as f64,
             100.0 * records.iter().map(|r| r.reasoning_tokens as f64 / r.completion_tokens.max(1) as f64).fold(f64::NEG_INFINITY, f64::max));

    let csv_path = format!("reasoning_{}_{}_{}.csv",
                           model.replace('-', "_"), max_completion_tokens, n_calls);
    let mut csv = File::create(&csv_path)?;
    writeln!(csv, "idx,reservation_nc,actual_nc,refund_nc,input_tokens,completion_tokens,reasoning_tokens,visible_output_tokens,latency_ms,margin_ratio,output_capped")?;
    for r in &records {
        writeln!(csv, "{},{},{},{},{},{},{},{},{},{:.6},{}",
                 r.idx, r.reservation_nc, r.actual_nc, r.refund_nc,
                 r.input_tokens, r.completion_tokens, r.reasoning_tokens,
                 r.visible_output_tokens, r.latency_ms, r.margin_ratio,
                 r.output_capped)?;
    }
    println!("Wrote {} rows to {}", records.len(), csv_path);
    Ok(())
}