//! Live-API evaluation of the receipt/refund discipline on Anthropic
//! Messages endpoint.
//!
//! Reviewer concern (Overclaim 3): "the receipt/refund mechanism is
//! described in §IV-F but has zero real-API evaluation." This binary
//! addresses that by running 10 real Anthropic calls through the
//! reserve-confirm-refund cycle and validating:
//!
//! 1. Every call has reservation ≥ actual (conservative-estimator
//!    discipline holds in vivo).
//! 2. Every refund is exactly reservation - actual (no rounding /
//!    truncation bugs in the arithmetic).
//! 3. Final budget equals B0 - sum(actual costs), conserving
//!    micro-cents exactly.
//! 4. Total refunded equals sum(reservation_i) - sum(actual_i)
//!    (the over-reservation overhead, measurable in vivo).
//!
//! ## Cost
//!
//! Roughly $0.10–$0.20 in Anthropic charges at current rates
//! (10 calls × ~$0.01–$0.02 each). The cap is set to $1.00 worth
//! of micro-cents ($MAX = 1_000_000$ uc), so even a 20× cost
//! increase from the estimate would not exhaust the budget.
//!
//! ## To run
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run --release --bin refund-live
//! ```
//!
//! Output: per-call line on stdout, summary CSV at
//! `refund_live_results.csv`.

use anyhow::{Context, Result};
use budget_typed_cap::Budget;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

/// Anthropic API rates (Claude Haiku 4.5, as of 2026-Q1).
const ANTHROPIC_PER_IN_TOKEN_UC: u64 = 1; // $1/M input tokens = 1 uc/token
const ANTHROPIC_PER_OUT_TOKEN_UC: u64 = 5; // $5/M output tokens = 5 uc/token

/// Budget cap = $1.00 = 1_000_000 micro-cents. Comfortably above
/// 10 × ~$0.02 per call = $0.20 expected total cost.
const BUDGET_CAP: u64 = 1_000_000;
type B = Budget<BUDGET_CAP>;

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const MODEL: &str = "claude-haiku-4-5-20251001";

/// Maximum output tokens to reserve per call. Conservative — the
/// prompts are simple Q&A, real outputs likely 50-150 tokens.
const MAX_OUT_TOKENS: u32 = 200;

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

/// Test prompts: short Q&A spanning different topics. The byte-length
/// estimator should be comfortably above the actual tokenizer cost
/// for all of these.
const PROMPTS: &[&str] = &[
    "What is the capital of France?",
    "Briefly explain why the sky is blue.",
    "List three primary colors.",
    "What is 17 times 23?",
    "Name two programming languages from the 1970s.",
    "Briefly describe what a haiku is.",
    "What is the speed of light in vacuum, approximately?",
    "Name one chemical element starting with 'N'.",
    "What year did the Berlin Wall fall?",
    "Briefly explain what an affine type is.",
];

#[derive(Debug, Serialize)]
struct CallRecord {
    prompt_idx: usize,
    prompt_byte_len: usize,
    reservation_uc: u64,
    actual_uc: u64,
    refund_uc: u64,
    input_tokens: u64,
    output_tokens: u64,
    budget_before: u64,
    budget_after: u64,
}

/// Compute conservative reservation from the request body byte
/// length and the max output tokens.
fn compute_reservation(request_body: &str) -> u64 {
    let byte_len = request_body.len() as u64;
    // Input cost: 1 uc per byte (conservative byte-length bound at $1/Mtok).
    let input_uc = byte_len * ANTHROPIC_PER_IN_TOKEN_UC;
    // Output cost: max_tokens × 5 uc/token.
    let output_uc = (MAX_OUT_TOKENS as u64) * ANTHROPIC_PER_OUT_TOKEN_UC;
    input_uc + output_uc
}

/// Compute actual cost from provider-reported token usage.
fn compute_actual(usage: &Usage) -> u64 {
    usage.input_tokens * ANTHROPIC_PER_IN_TOKEN_UC
        + usage.output_tokens * ANTHROPIC_PER_OUT_TOKEN_UC
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY must be set")?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    let mut budget: Option<B> = Some(Budget::new(BUDGET_CAP)?);
    let mut records: Vec<CallRecord> = Vec::with_capacity(PROMPTS.len());
    let mut sum_reserved = 0u64;
    let mut sum_actual = 0u64;
    let mut sum_refund = 0u64;

    println!(
        "Initial budget: {} uc (${:.6})",
        BUDGET_CAP,
        BUDGET_CAP as f64 / 1_000_000.0
    );
    println!(
        "{:<3}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  Prompt",
        "#", "reserve", "actual", "refund", "in_tok", "out_tok"
    );

    for (i, prompt) in PROMPTS.iter().enumerate() {
        let req = AnthropicReq {
            model: MODEL,
            max_tokens: MAX_OUT_TOKENS,
            messages: vec![Msg { role: "user", content: prompt }],
        };
        let body = serde_json::to_string(&req)?;
        let byte_len = body.len();
        let reservation = compute_reservation(&body);

        let current_budget = budget.take().expect("budget should be present");
        let budget_before = current_budget.micro_cents();

        // Step 1: reserve.
        let (budget_after_reserve, receipt) = current_budget
            .spend_with_receipt(reservation)
            .with_context(|| format!("call {} reserve failed", i))?;

        // Step 2: make the API call.
        let resp = client
            .post(ANTHROPIC_URL)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API returned {}: {}", status, txt);
        }
        let parsed: AnthropicResp = resp.json().await?;
        let actual = compute_actual(&parsed.usage);

        // Step 3: confirm + refund.
        let refund = receipt
            .confirm(actual)
            .with_context(|| {
                format!(
                    "call {} confirm failed: actual={} > reserved={}",
                    i, actual, reservation
                )
            })?;
        let refund_amount = refund.amount();
        let budget_after = refund.apply_to(budget_after_reserve)?;
        let budget_after_value = budget_after.micro_cents();

        records.push(CallRecord {
            prompt_idx: i,
            prompt_byte_len: byte_len,
            reservation_uc: reservation,
            actual_uc: actual,
            refund_uc: refund_amount,
            input_tokens: parsed.usage.input_tokens,
            output_tokens: parsed.usage.output_tokens,
            budget_before,
            budget_after: budget_after_value,
        });

        sum_reserved += reservation;
        sum_actual += actual;
        sum_refund += refund_amount;

        println!(
            "{:<3}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {}",
            i,
            reservation,
            actual,
            refund_amount,
            parsed.usage.input_tokens,
            parsed.usage.output_tokens,
            prompt.chars().take(48).collect::<String>(),
        );

        budget = Some(budget_after);
    }

    let final_budget = budget.unwrap();
    let final_value = final_budget.micro_cents();

    println!();
    println!("=== Summary ===");
    println!("Initial budget:        {:>10} uc  (${:.6})", BUDGET_CAP, BUDGET_CAP as f64 / 1e6);
    println!("Total reserved:        {:>10} uc  (${:.6})", sum_reserved, sum_reserved as f64 / 1e6);
    println!("Total actual cost:     {:>10} uc  (${:.6})", sum_actual, sum_actual as f64 / 1e6);
    println!("Total refunded:        {:>10} uc  (${:.6})", sum_refund, sum_refund as f64 / 1e6);
    println!("Final budget:          {:>10} uc  (${:.6})", final_value, final_value as f64 / 1e6);

    let over_reservation_pct = if sum_actual > 0 {
        100.0 * (sum_reserved as f64 - sum_actual as f64) / sum_actual as f64
    } else {
        0.0
    };
    println!("Over-reservation:      {:.1}%", over_reservation_pct);

    // Conservation check: initial - actual == final budget.
    let expected_final = BUDGET_CAP - sum_actual;
    println!();
    if final_value == expected_final {
        println!(
            "✓ Conservation verified: initial ({}) - actual ({}) = final ({})",
            BUDGET_CAP, sum_actual, final_value
        );
    } else {
        println!(
            "✗ CONSERVATION VIOLATED: initial ({}) - actual ({}) = {} but final = {}",
            BUDGET_CAP, sum_actual, expected_final, final_value
        );
        anyhow::bail!("conservation violated");
    }

    // A1 in vivo: every call had reservation ≥ actual.
    let a1_violations: Vec<_> = records
        .iter()
        .filter(|r| r.actual_uc > r.reservation_uc)
        .collect();
    if a1_violations.is_empty() {
        println!(
            "✓ A1 holds: every call had reservation ≥ actual ({} / {} calls)",
            records.len(),
            records.len()
        );
    } else {
        println!(
            "✗ A1 VIOLATED on {} of {} calls",
            a1_violations.len(),
            records.len()
        );
        for r in &a1_violations {
            println!(
                "  call {}: reservation={}  actual={}",
                r.prompt_idx, r.reservation_uc, r.actual_uc
            );
        }
    }

    // Refund arithmetic: refund_i = reservation_i - actual_i for every i.
    let refund_errors: Vec<_> = records
        .iter()
        .filter(|r| r.refund_uc != r.reservation_uc - r.actual_uc)
        .collect();
    if refund_errors.is_empty() {
        println!(
            "✓ Refund arithmetic: refund == reservation − actual on every call ({} / {})",
            records.len(),
            records.len()
        );
    } else {
        println!(
            "✗ REFUND ARITHMETIC BROKEN on {} calls",
            refund_errors.len()
        );
    }

    // Write CSV.
    let mut csv = File::create("refund_live_results.csv")?;
    writeln!(
        csv,
        "prompt_idx,prompt_byte_len,reservation_uc,actual_uc,refund_uc,input_tokens,output_tokens,budget_before,budget_after"
    )?;
    for r in &records {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{}",
            r.prompt_idx,
            r.prompt_byte_len,
            r.reservation_uc,
            r.actual_uc,
            r.refund_uc,
            r.input_tokens,
            r.output_tokens,
            r.budget_before,
            r.budget_after,
        )?;
    }
    println!();
    println!("Wrote {} rows to refund_live_results.csv", records.len());

    Ok(())
}
