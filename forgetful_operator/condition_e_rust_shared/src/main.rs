use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Parser)]
#[command(
    name = "condition_e_rust_shared",
    about = "M7: Rust shared Arc<Mutex<Budget>> pre-flight reservation"
)]

struct Args {
    #[arg(long, default_value_t = 30)]
    trials: u32,

    #[arg(long, default_value_t = 60)]
    budget: u64,

    #[arg(long, default_value_t = 3)]
    children: u32,

    #[arg(long, default_value = "condition_e_results.csv")]
    output: std::path::PathBuf,

    #[arg(long, default_value = "claude-haiku-4-5")]
    model: String,

    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long, default_value_t = 30)]
    max_output_tokens: u32,

    #[arg(long, default_value_t = 31)]
    per_child_estimate: u64,
}

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

const INPUT_PRICE_UC_PER_TOKEN: u64 = 100; // $1/Mtok = 100 uc/Mtok = 0.0001 uc/token
const OUTPUT_PRICE_UC_PER_TOKEN: u64 = 500; // $5/Mtok = 500 uc/Mtok = 0.0005 uc/token

fn cost_uc(input_tokens: u64, output_tokens: u64) -> u64 {
    (input_tokens * INPUT_PRICE_UC_PER_TOKEN + output_tokens * OUTPUT_PRICE_UC_PER_TOKEN)
        / 1_000_000
        + 1
}

#[derive(Debug, Serialize)]
struct TrialResult {
    trial_id: u32,
    child_id: u32,
    outcome: String,
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

    let mut wtr = csv::Writer::from_path(&args.output)?;

    for r in &all_results {
        wtr.serialize(r)?;
    }

    wtr.flush()?;

    let overshoots = per_trial_total.iter().filter(|t| **t > args.budget).count();
    let mean_spend: f64 = per_trial_total.iter().sum::<u64>() as f64 / per_trial_total.len().max(1) as f64;

    eprintln!();
    eprintln!(" CONDITION E SUMMARY ");
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
