use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::time::Instant;
use tokio::task::JoinHandle;

use token_budgets::{Budget, BudgetMint};

type B = Budget<10_000>;

const LANG001_SYSTEM: &str =
    "You are a SQL agent. Use the provided sql_query tool to answer \
     the user's question. If the first tool call does not return the \
     expected information, try variations.";

const LANG001_USER: &str =
    "How many users registered in 2024 from the marketing campaign \
     table? The table schema is: users(id, email, signup_source, \
     signup_date). Use the sql_query tool.";

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 30)]
    n: usize,

    #[arg(long, default_value_t = 60)]
    cap: u64,

    #[arg(long, default_value = "results/rust_affine_anthropic.csv")]
    output: PathBuf,

    #[arg(long, default_value_t = 1.0)]
    rate_in: f64,

    #[arg(long, default_value_t = 5.0)]
    rate_out: f64,

    #[arg(long, default_value_t = 30)]
    max_output_tokens: u64,

    #[arg(long, default_value_t = 0.5)]
    margin: f64,
}

#[derive(Debug, Serialize)]
struct TrialResult {
    trial_id: usize,
    cap_uc: u64,
    total_spent_uc: u64,
    overshoot: u8,
    children_admitted: u8,
    children_completed: u8,
    elapsed_s: f64,
    error: String,
}

#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    usage: AnthropicUsage,
}

fn estimate_uc_byte_length(
    prompt: &str,
    max_output_tokens: u64,
    rate_in_per_mtok: f64,
    rate_out_per_mtok: f64,
    margin: f64,
) -> u64 {
    let input_tokens_est = (margin * prompt.len() as f64) as u64;
    let in_uc = ((input_tokens_est as f64) * rate_in_per_mtok / 10.0) as u64;
    let out_uc = ((max_output_tokens as f64) * rate_out_per_mtok / 10.0) as u64;
    in_uc + out_uc
}

async fn call_anthropic(
    client: &reqwest::Client,
    api_key: &str,
    max_output_tokens: u64,
) -> Result<AnthropicResponse> {
    let body = serde_json::json!({
        "model": "claude-haiku-4-5",
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "system": LANG001_SYSTEM,
        "messages": [{"role": "user", "content": LANG001_USER}],
    });

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("anthropic request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("anthropic returned {}: {}", status, text);
    }

    resp.json::<AnthropicResponse>()
        .await
        .context("parsing anthropic response")
}

async fn child(
    budget: B,
    api_key: String,
    client: reqwest::Client,
    rate_in: f64,
    rate_out: f64,
    max_output_tokens: u64,
    margin: f64,
) -> Result<(u64, bool)> {
    let estimate = estimate_uc_byte_length(
        &format!("{}{}", LANG001_SYSTEM, LANG001_USER),
        max_output_tokens,
        rate_in,
        rate_out,
        margin,
    );

    let _remainder = match budget.spend(estimate) {
        Ok(b) => b,
        Err(_) => return Ok((0, false)), // refused at pre-flight
    };

    let response = call_anthropic(&client, &api_key, max_output_tokens).await?;
    let actual_uc = (response.usage.input_tokens as f64 * rate_in / 10.0
        + response.usage.output_tokens as f64 * rate_out / 10.0)
        as u64;

    Ok((actual_uc, true))
}

async fn run_trial(
    trial_id: usize,
    cap_uc: u64,
    mint: &BudgetMint,
    api_key: String,
    client: reqwest::Client,
    rate_in: f64,
    rate_out: f64,
    max_output_tokens: u64,
    margin: f64,
) -> TrialResult {
    let t0 = Instant::now();

    // Mint the parent budget via the capability gate.
    let parent_budget = match B::mint(mint, cap_uc) {
        Ok(b) => b,
        Err(e) => {
            return TrialResult {
                trial_id, cap_uc, total_spent_uc: 0, overshoot: 0,
                children_admitted: 0, children_completed: 0,
                elapsed_s: t0.elapsed().as_secs_f64(),
                error: format!("mint failed: {:?}", e),
            };
        }
    };

    let per_child = cap_uc / 3;
    let (b1, rest_after_first) = match parent_budget.split(per_child) {
        Ok(t) => t,
        Err(e) => {
            return TrialResult {
                trial_id, cap_uc, total_spent_uc: 0, overshoot: 0,
                children_admitted: 0, children_completed: 0,
                elapsed_s: t0.elapsed().as_secs_f64(),
                error: format!("first split failed: {:?}", e),
            };
        }
    };

    let (b2, b3) = match rest_after_first.split(per_child) {
        Ok(t) => t,
        Err(e) => {
            return TrialResult {
                trial_id, cap_uc, total_spent_uc: 0, overshoot: 0,
                children_admitted: 0, children_completed: 0,
                elapsed_s: t0.elapsed().as_secs_f64(),
                error: format!("second split failed: {:?}", e),
            };
        }
    };

    let h1: JoinHandle<Result<(u64, bool)>> = tokio::spawn(child(
        b1, api_key.clone(), client.clone(),
        rate_in, rate_out, max_output_tokens, margin,
    ));
    let h2: JoinHandle<Result<(u64, bool)>> = tokio::spawn(child(
        b2, api_key.clone(), client.clone(),
        rate_in, rate_out, max_output_tokens, margin,
    ));
    let h3: JoinHandle<Result<(u64, bool)>> = tokio::spawn(child(
        b3, api_key.clone(), client.clone(),
        rate_in, rate_out, max_output_tokens, margin,
    ));

    let r1 = h1.await;
    let r2 = h2.await;
    let r3 = h3.await;

    let results: Vec<Result<(u64, bool)>> = vec![
        r1.unwrap_or_else(|e| Err(anyhow::anyhow!("h1 join: {:?}", e))),
        r2.unwrap_or_else(|e| Err(anyhow::anyhow!("h2 join: {:?}", e))),
        r3.unwrap_or_else(|e| Err(anyhow::anyhow!("h3 join: {:?}", e))),
    ];

    let mut total_spent = 0u64;
    let mut admitted = 0u8;
    let mut completed = 0u8;
    let mut errors: Vec<String> = Vec::new();

    for r in results {
        match r {
            Ok((cost, was_admitted)) => {
                if was_admitted {
                    admitted += 1;
                    completed += 1;
                    total_spent += cost;
                }
            }
            Err(e) => {
                errors.push(format!("{:?}", e));
            }
        }
    }

    TrialResult {
        trial_id,
        cap_uc,
        total_spent_uc: total_spent,
        overshoot: if total_spent > cap_uc { 1 } else { 0 },
        children_admitted: admitted,
        children_completed: completed,
        elapsed_s: t0.elapsed().as_secs_f64(),
        error: errors.join("; "),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let api_key = env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY env var not set")?;

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mint = BudgetMint::take_authority();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    let estimate = estimate_uc_byte_length(
        &format!("{}{}", LANG001_SYSTEM, LANG001_USER),
        args.max_output_tokens, args.rate_in, args.rate_out, args.margin,
    );
    let sub_budget = args.cap / 3;

    println!(
        "Running rust_affine_split: N={}, cap={} uc",
        args.n, args.cap
    );
    println!(
        "  estimate per child = {} uc, sub-budget per child = {} uc",
        estimate, sub_budget
    );
    println!(
        "  pre-flight check inside spawned task: estimate <= sub-budget? {} ({} <= {})",
        estimate <= sub_budget, estimate, sub_budget
    );
    if estimate > sub_budget {
        println!("  (all children will be refused; this is the discipline's");
        println!("   refusal-to-operate regime, structurally correct)");
    }

    let mut wtr = csv::Writer::from_path(&args.output)?;
    let mut overshoots = 0;

    for i in 0..args.n {
        let r = run_trial(
            i, args.cap, &mint, api_key.clone(), client.clone(),
            args.rate_in, args.rate_out,
            args.max_output_tokens, args.margin,
        )
            .await;

        let err_summary = if r.error.is_empty() {
            String::new()
        } else {
            format!(", ERR: {}", r.error)
        };
        println!(
            "  trial {}: spent={} uc, overshoot={}, admitted={}/3, elapsed={:.1}s{}",
            r.trial_id, r.total_spent_uc, r.overshoot,
            r.children_admitted, r.elapsed_s, err_summary
        );
        if r.overshoot == 1 {
            overshoots += 1;
        }
        wtr.serialize(&r)?;
    }
    wtr.flush()?;

    println!("\nSUMMARY: {}/{} overshoots", overshoots, args.n);
    println!("Output: {}", args.output.display());

    Ok(())
}