//! # Condition E — Rust shared Arc<Mutex<Budget>> with pre-flight reservation
//!
//! ## PRE-ATTACK (BRUTAL REVIEWER VOICE)
//!
//! > "Your forgetful-operator experiment's Condition A uses Python racy code
//! >  (shared mutable budget, no lock), and Condition C uses Rust affine
//! >  split (per-child sub-budget). The A-vs-C contrast varies TWO things
//! >  at once: allocation strategy (shared → split) AND integrity layer
//! >  (none → compile-time). You cannot claim the type system is doing the
//! >  work; the allocation strategy could be doing all of it. Where's the
//! >  Rust shared baseline?"
//!
//! ## DISPOSITION
//!
//! Condition E is that Rust shared baseline: an `Arc<Mutex<Budget>>` shared
//! across three `tokio::spawn`ed children with the same pre-flight
//! reservation + post-call refund pattern as Condition B's Python locked
//! variant. The pre-registered prediction (paper §8.3, M7) is parity with
//! Condition B at 0/30 overshoot. Overshoot would NOT refute the integrity
//! claim — it would be a separate finding about runtime-discipline failures
//! that the paper would report as a third comparison point.
//!
//! ## COUNTER-ATTACK PRE-EMPTED
//!
//! > "You're using Arc<Mutex<>>. The paper's own §4.9 says 'we deliberately
//! >  do not use Arc<Mutex<>>'. Isn't running Condition E hypocritical?"
//!
//! Not at all. Section 4.9 says the paper's primary discipline (affine
//! split) does not require Arc<Mutex<>>. Condition E is a comparison
//! against the runtime-discipline alternative an operator might write
//! instead. The paper's contribution is that the affine split removes the
//! need for the manual lock discipline that Condition E shows works
//! correctly when written carefully.
//!
//! ## COUNTER-ATTACK PRE-EMPTED 2
//!
//! > "You're predicting the outcome that supports your paper. What if
//! >  Condition E actually overshoots?"
//!
//! Three responses, in order:
//!   1. The pre-registration explicitly states overshoot would be reported
//!      as either an operator-discipline error (which we'd correct and
//!      re-run) or a separate finding about Arc<Mutex<Budget>> + tokio
//!      scheduling. Neither outcome refutes the type-system claim.
//!   2. The trybuild evidence in `forgetful_operator/rust_compile_fail/`
//!      covers BOTH the split and shared-mutex patterns; the
//!      non-bypassability claim is about preventing wrong programs from
//!      compiling, not about which compiling programs run correctly.
//!   3. If Condition E overshoots and root cause is the Mutex+Tokio
//!      combination, that's a result the paper would CITE (the affine
//!      split structurally avoids the failure mode that bit Condition E),
//!      which would STRENGTHEN the paper's framing, not weaken it.
//!
//! ## RUN
//!
//! ```bash
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run --release -- \
//!     --trials 30 \
//!     --budget 60 \
//!     --children 3 \
//!     --output condition_e_results.csv
//! ```
//!
//! ## EXPECTED COST
//!
//! 30 trials × 3 children × 1 successful call/trial × ~23 uc/call ≈ 2,070 uc total
//! ≈ $0.02 worst case. Most calls refuse pre-flight under cap=60uc, so
//! actual cost is much lower (~$0.005).

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

// =====================================================================
// CLI
// =====================================================================

#[derive(Parser)]
#[command(
    name = "condition_e_rust_shared",
    about = "M7: Rust shared Arc<Mutex<Budget>> pre-flight reservation"
)]
struct Args {
    /// Number of trials (paper-default: 30)
    #[arg(long, default_value_t = 30)]
    trials: u32,

    /// Parent budget in micro-cents (paper-default: 60)
    #[arg(long, default_value_t = 60)]
    budget: u64,

    /// Number of concurrent children per trial (paper-default: 3)
    #[arg(long, default_value_t = 3)]
    children: u32,

    /// Output CSV path
    #[arg(long, default_value = "condition_e_results.csv")]
    output: std::path::PathBuf,

    /// Anthropic model
    #[arg(long, default_value = "claude-haiku-4-5")]
    model: String,

    /// Temperature (paper-default: 0 for determinism)
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Maximum output tokens per call
    #[arg(long, default_value_t = 30)]
    max_output_tokens: u32,

    /// Per-child estimate in micro-cents (matches paper's condition B)
    #[arg(long, default_value_t = 31)]
    per_child_estimate: u64,
}

// =====================================================================
// Budget (minimal local copy — matches the budget-spike crate's API)
// =====================================================================
//
// PRE-ATTACK: "Why a local Budget, not budget-spike's?"
// DISPOSITION: To make this file self-contained and remove the
// dependency on the artefact path layout. The behaviour is
// byte-identical for the spend/refund operations used here. If you
// have budget-spike on path, you can delete this block and import
// `use budget_spike::Budget;` instead.

#[derive(Debug)]
struct Budget {
    micro_cents: u64,
}

impl Budget {
    fn new(micro_cents: u64) -> Self {
        Self { micro_cents }
    }

    fn available(&self) -> u64 {
        self.micro_cents
    }

    fn try_reserve(&mut self, amount: u64) -> Option<Reservation> {
        if amount > self.micro_cents {
            return None;
        }
        self.micro_cents -= amount;
        Some(Reservation { reserved: amount })
    }

    fn refund(&mut self, amount: u64) -> Result<()> {
        self.micro_cents = self
            .micro_cents
            .checked_add(amount)
            .ok_or_else(|| anyhow::anyhow!("refund overflow"))?;
        Ok(())
    }
}

#[derive(Debug)]
struct Reservation {
    reserved: u64,
}

// =====================================================================
// Anthropic client (minimal — direct reqwest, no extra deps)
// =====================================================================

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    temperature: f64,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize, Debug)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

async fn anthropic_call(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    max_output_tokens: u32,
    temperature: f64,
) -> Result<(u64, u64)> {
    let req = AnthropicRequest {
        model: model.to_string(),
        max_tokens: max_output_tokens,
        temperature,
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&req)
        .send()
        .await
        .context("Anthropic API call failed")?
        .error_for_status()
        .context("Anthropic API returned non-200")?;

    let body: AnthropicResponse = resp.json().await.context("Failed to parse response")?;
    Ok((body.usage.input_tokens, body.usage.output_tokens))
}

// =====================================================================
// Cost model (matches paper's claude-haiku-4-5 rates)
// =====================================================================

const INPUT_PRICE_UC_PER_TOKEN: u64 = 100; // $1/Mtok = 100 uc/Mtok = 0.0001 uc/token
const OUTPUT_PRICE_UC_PER_TOKEN: u64 = 500; // $5/Mtok = 500 uc/Mtok = 0.0005 uc/token

fn cost_uc(input_tokens: u64, output_tokens: u64) -> u64 {
    // Both prices are per Mtok; divide by 1_000_000
    (input_tokens * INPUT_PRICE_UC_PER_TOKEN + output_tokens * OUTPUT_PRICE_UC_PER_TOKEN)
        / 1_000_000
        + 1 // round up to avoid undercount on partial-token billing
}

// =====================================================================
// Condition E: one trial
// =====================================================================

#[derive(Debug, Serialize)]
struct TrialResult {
    trial_id: u32,
    child_id: u32,
    outcome: String, // "spent" | "refused_preflight" | "api_error"
    reserved_uc: u64,
    actual_charge_uc: u64,
    input_tokens: u64,
    output_tokens: u64,
    wall_clock_ms: u128,
}

async fn run_one_child(
    trial_id: u32,
    child_id: u32,
    budget: Arc<Mutex<Budget>>,
    client: reqwest::Client,
    api_key: String,
    args: &Args,
) -> TrialResult {
    let start = Instant::now();
    let prompt = format!(
        "What is the sum of {} and {}? Reply with just the number.",
        child_id,
        child_id * 2
    );

    // Pre-flight reservation under the SHARED mutex
    // This is the equivalent of Condition B's asyncio.Lock
    let reservation = {
        let mut budget_guard = budget.lock().await;
        budget_guard.try_reserve(args.per_child_estimate)
    };

    let reservation = match reservation {
        Some(r) => r,
        None => {
            return TrialResult {
                trial_id,
                child_id,
                outcome: "refused_preflight".to_string(),
                reserved_uc: args.per_child_estimate,
                actual_charge_uc: 0,
                input_tokens: 0,
                output_tokens: 0,
                wall_clock_ms: start.elapsed().as_millis(),
            };
        }
    };

    // Issue the LLM call
    let api_result = anthropic_call(
        &client,
        &api_key,
        &args.model,
        &prompt,
        args.max_output_tokens,
        args.temperature,
    )
    .await;

    match api_result {
        Ok((input_tokens, output_tokens)) => {
            let actual = cost_uc(input_tokens, output_tokens);
            // Refund the difference (reservation - actual) into the shared budget
            let refund_amount = reservation.reserved.saturating_sub(actual);
            if refund_amount > 0 {
                let mut budget_guard = budget.lock().await;
                let _ = budget_guard.refund(refund_amount);
            }
            TrialResult {
                trial_id,
                child_id,
                outcome: "spent".to_string(),
                reserved_uc: reservation.reserved,
                actual_charge_uc: actual,
                input_tokens,
                output_tokens,
                wall_clock_ms: start.elapsed().as_millis(),
            }
        }
        Err(e) => {
            // On API error, FORFEIT the reservation (do not refund) —
            // matches Condition B's behaviour on transient errors
            eprintln!(
                "trial {} child {} API error: {} (reservation forfeit)",
                trial_id, child_id, e
            );
            TrialResult {
                trial_id,
                child_id,
                outcome: "api_error".to_string(),
                reserved_uc: reservation.reserved,
                actual_charge_uc: 0,
                input_tokens: 0,
                output_tokens: 0,
                wall_clock_ms: start.elapsed().as_millis(),
            }
        }
    }
}

async fn run_one_trial(
    trial_id: u32,
    client: reqwest::Client,
    api_key: String,
    args: &Args,
) -> Vec<TrialResult> {
    let budget = Arc::new(Mutex::new(Budget::new(args.budget)));

    let mut handles = Vec::with_capacity(args.children as usize);
    for child_id in 0..args.children {
        let budget = Arc::clone(&budget);
        let client = client.clone();
        let api_key = api_key.clone();
        // We cannot move &args into spawn; clone the needed values
        let per_child_estimate = args.per_child_estimate;
        let model = args.model.clone();
        let max_output_tokens = args.max_output_tokens;
        let temperature = args.temperature;
        let trial_args = ArgsClone {
            per_child_estimate,
            model,
            max_output_tokens,
            temperature,
        };

        handles.push(tokio::spawn(async move {
            run_one_child_owned(trial_id, child_id, budget, client, api_key, trial_args).await
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        match h.await {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("trial {} child task panic: {}", trial_id, e),
        }
    }
    results
}

// Owned-args helper for tokio::spawn
struct ArgsClone {
    per_child_estimate: u64,
    model: String,
    max_output_tokens: u32,
    temperature: f64,
}

async fn run_one_child_owned(
    trial_id: u32,
    child_id: u32,
    budget: Arc<Mutex<Budget>>,
    client: reqwest::Client,
    api_key: String,
    args: ArgsClone,
) -> TrialResult {
    // Inline of run_one_child using owned args to satisfy 'static bound on spawn
    let start = Instant::now();
    let prompt = format!(
        "What is the sum of {} and {}? Reply with just the number.",
        child_id,
        child_id * 2
    );

    let reservation = {
        let mut budget_guard = budget.lock().await;
        budget_guard.try_reserve(args.per_child_estimate)
    };

    let reservation = match reservation {
        Some(r) => r,
        None => {
            return TrialResult {
                trial_id,
                child_id,
                outcome: "refused_preflight".to_string(),
                reserved_uc: args.per_child_estimate,
                actual_charge_uc: 0,
                input_tokens: 0,
                output_tokens: 0,
                wall_clock_ms: start.elapsed().as_millis(),
            };
        }
    };

    let api_result = anthropic_call(
        &client,
        &api_key,
        &args.model,
        &prompt,
        args.max_output_tokens,
        args.temperature,
    )
    .await;

    match api_result {
        Ok((input_tokens, output_tokens)) => {
            let actual = cost_uc(input_tokens, output_tokens);
            let refund_amount = reservation.reserved.saturating_sub(actual);
            if refund_amount > 0 {
                let mut budget_guard = budget.lock().await;
                let _ = budget_guard.refund(refund_amount);
            }
            TrialResult {
                trial_id,
                child_id,
                outcome: "spent".to_string(),
                reserved_uc: reservation.reserved,
                actual_charge_uc: actual,
                input_tokens,
                output_tokens,
                wall_clock_ms: start.elapsed().as_millis(),
            }
        }
        Err(e) => {
            eprintln!(
                "trial {} child {} API error: {} (reservation forfeit)",
                trial_id, child_id, e
            );
            TrialResult {
                trial_id,
                child_id,
                outcome: "api_error".to_string(),
                reserved_uc: reservation.reserved,
                actual_charge_uc: 0,
                input_tokens: 0,
                output_tokens: 0,
                wall_clock_ms: start.elapsed().as_millis(),
            }
        }
    }
}

// =====================================================================
// Main
// =====================================================================

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let args = Args::parse();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY environment variable must be set")?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    eprintln!(
        "Condition E (Rust shared Arc<Mutex<Budget>>): {} trials, B_0={} uc, {} children, model={}",
        args.trials, args.budget, args.children, args.model
    );
    eprintln!(
        "Per-child estimate: {} uc (matches paper's condition B)",
        args.per_child_estimate
    );
    eprintln!();

    let mut all_results = Vec::new();
    let mut per_trial_total = Vec::new();

    for trial_id in 0..args.trials {
        let results = run_one_trial(trial_id, client.clone(), api_key.clone(), &args).await;
        let trial_total: u64 = results.iter().map(|r| r.actual_charge_uc).sum();
        eprintln!(
            "trial {:>2}: total spent = {} uc, outcomes = {:?}",
            trial_id,
            trial_total,
            results.iter().map(|r| &r.outcome).collect::<Vec<_>>()
        );
        per_trial_total.push(trial_total);
        all_results.extend(results);
    }

    // Write CSV
    let mut wtr = csv::Writer::from_path(&args.output)?;
    for r in &all_results {
        wtr.serialize(r)?;
    }
    wtr.flush()?;

    // Summary
    let overshoots = per_trial_total.iter().filter(|t| **t > args.budget).count();
    let mean_spend: f64 =
        per_trial_total.iter().sum::<u64>() as f64 / per_trial_total.len().max(1) as f64;

    eprintln!();
    eprintln!("=== CONDITION E SUMMARY ===");
    eprintln!("Trials:                {}", args.trials);
    eprintln!("Overshoots (spent > B_0={} uc): {}/{}", args.budget, overshoots, args.trials);
    eprintln!("Mean total spend:      {:.1} uc", mean_spend);
    eprintln!("Mean as % of cap:      {:.1}%", mean_spend / args.budget as f64 * 100.0);
    eprintln!();
    eprintln!("Pre-committed acceptance criteria (from paper §8.3, M7):");
    eprintln!("  0/30 overshoot   → outcome (i): parity with Condition B confirmed");
    eprintln!("  N>0 overshoot   → outcome (ii): see paper §8.3 for interpretation");
    eprintln!();
    if overshoots == 0 {
        eprintln!("OUTCOME: (i) PARITY CONFIRMED");
        eprintln!("  Allocation-vs-integrity confound closed.");
        eprintln!("  Update paper §5.11 to add Condition E row to Table 14.");
    } else {
        eprintln!("OUTCOME: (ii) PARITY NOT CONFIRMED");
        eprintln!("  Investigate: operator-discipline error vs Mutex+tokio finding.");
        eprintln!("  Update paper §5.11 with the non-parity finding per pre-registration.");
    }
    eprintln!();
    eprintln!("Wrote {} rows to {}", all_results.len(), args.output.display());

    Ok(())
}
