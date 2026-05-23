//! reasoning_eval_harness.rs
//!
//! Live-API evaluation of `Budget::spend_with_reasoning` +
//! `ReasoningProvider` against actual reasoning models.
//!
//! Goal: empirically validate that the session-cumulative budgeting
//! discipline holds when stacked on top of provider-side per-call
//! reasoning controls. We exercise three configurations:
//!
//!   (A) Anthropic Sonnet with extended thinking enabled
//!       (thinking.budget_tokens parameter)
//!   (B) OpenAI o4-mini with reasoning_effort=medium
//!       (reasoning_effort parameter)
//!   (C) DeepSeek R1 (open-weight reasoning model)
//!
//! For each configuration, we run N=20 trials of a multi-step
//! reasoning workload, with the session cap = $0.30 (300000 uc) and
//! per-call max_output_tokens = 500 + provider-specific reasoning
//! reservation (1000 uc for Anthropic, 1500 uc for OpenAI o-series,
//! 1000 uc for DeepSeek).
//!
//! Place this file at:
//!   token-budgets-experiments/budget-spike/src/bin/reasoning_eval.rs
//!
//! Add to Cargo.toml [dependencies]:
//!   reqwest = { version = "0.12", features = ["json"] }
//!   tokio = { version = "1", features = ["full"] }
//!   serde = { version = "1", features = ["derive"] }
//!   serde_json = "1"
//!   csv = "1"
//!   anyhow = "1"
//!
//! Build and run:
//!   cargo build --release --bin reasoning_eval
//!   export ANTHROPIC_API_KEY=...
//!   export OPENAI_API_KEY=...
//!   export DEEPSEEK_API_KEY=...
//!   cargo run --release --bin reasoning_eval -- --provider anthropic --n 20
//!   cargo run --release --bin reasoning_eval -- --provider openai --n 20
//!   cargo run --release --bin reasoning_eval -- --provider deepseek --n 20
//!
//! Output CSVs (placed in multiway/sweep_results/):
//!   reasoning_eval_anthropic_thinking_n20.csv
//!   reasoning_eval_openai_o4mini_n20.csv
//!   reasoning_eval_deepseek_r1_n20.csv
//!
//! Cost estimate per provider (rough):
//!   Anthropic Sonnet w/ thinking: $3 in + $15 out + $15 thinking per Mtok
//!     ~$2.50 for 20 runs * 5 calls/run
//!   OpenAI o4-mini: $0.55 in + $4.40 out per Mtok
//!     ~$0.60 for 20 runs * 5 calls/run
//!   DeepSeek R1: $0.55 in + $2.19 out per Mtok
//!     ~$0.40 for 20 runs * 5 calls/run
//! Budget $5 for all three.

use anyhow::{anyhow, Context, Result};
use budget_spike::{Budget, BudgetMint, ReasoningProvider};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

const SONNET_MODEL: &str = "claude-sonnet-4-5-20250929";
const O4_MINI_MODEL: &str = "o4-mini";
const DEEPSEEK_MODEL: &str = "deepseek-reasoner";

const CAP_UC: u64 = 300_000; // $0.30 session cap
const MAX_OUTPUT_TOKENS: u32 = 500;
const MAX_STEPS_PER_TRIAL: usize = 8;
const OVERLOAD_BACKOFF_S: &[u64] = &[2, 5, 10, 20, 30];

// Per-call p99 reasoning reservation in micro-cents.
// These are conservative reservations calibrated against publicly
// reported p99 reasoning-token usage for the respective providers.
const ANTHROPIC_REASONING_P99_UC: u64 = 1500;
const OPENAI_O_SERIES_REASONING_P99_UC: u64 = 2000;
const DEEPSEEK_R1_REASONING_P99_UC: u64 = 1200;

const REASONING_WORKLOAD_SYSTEM: &str = "You are a step-by-step reasoning assistant. Show your reasoning explicitly. After your thinking, give a concise answer.";

const REASONING_WORKLOAD_USER: &str = "A train leaves Station A at 9:00 AM travelling at 80 km/h. Another train leaves Station B (240 km away) at 9:30 AM travelling toward A at 100 km/h. At what time and how far from Station A do they meet? Show all work.";

#[derive(Debug, Serialize)]
struct TrialResult {
    trial_id: usize,
    provider_label: String,
    steps: usize,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_reasoning_tokens: u64,
    total_billed_uc: u64,
    total_reserved_uc: u64,
    overshoot: u8,
    refused_at_step: i32, // -1 if not refused
    refused_reason: String,
    retries_total: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    ty: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

enum ProviderConfig {
    Anthropic,
    OpenAI,
    DeepSeek,
}

impl ProviderConfig {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAI),
            "deepseek" => Ok(Self::DeepSeek),
            other => Err(anyhow!("unknown provider: {}", other)),
        }
    }

    fn reasoning_provider(&self) -> ReasoningProvider {
        match self {
            Self::Anthropic => ReasoningProvider::Anthropic,
            Self::OpenAI => ReasoningProvider::OpenAIO1 {
                per_call_reasoning_p99_uc: OPENAI_O_SERIES_REASONING_P99_UC,
            },
            Self::DeepSeek => ReasoningProvider::DeepSeekR1 {
                per_call_reasoning_p99_uc: DEEPSEEK_R1_REASONING_P99_UC,
            },
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic_sonnet_thinking",
            Self::OpenAI => "openai_o4mini_medium",
            Self::DeepSeek => "deepseek_r1",
        }
    }

    fn output_filename(&self) -> &'static str {
        match self {
            Self::Anthropic => "reasoning_eval_anthropic_thinking_n20.csv",
            Self::OpenAI => "reasoning_eval_openai_o4mini_n20.csv",
            Self::DeepSeek => "reasoning_eval_deepseek_r1_n20.csv",
        }
    }
}

// Per-token billing rates in micro-cents per token.
struct Rates {
    input_per_tok_uc: u64,
    output_per_tok_uc: u64,
    reasoning_per_tok_uc: u64,
}

impl Rates {
    fn for_provider(p: &ProviderConfig) -> Self {
        match p {
            // Sonnet: $3 / $15 / $15 per Mtok (input / output / thinking)
            ProviderConfig::Anthropic => Rates {
                input_per_tok_uc: 3,
                output_per_tok_uc: 15,
                reasoning_per_tok_uc: 15,
            },
            // o4-mini: $0.55 / $4.40 per Mtok (reasoning billed at output rate)
            ProviderConfig::OpenAI => Rates {
                input_per_tok_uc: 1, // rounded up; actual 0.55
                output_per_tok_uc: 5, // rounded up; actual 4.40
                reasoning_per_tok_uc: 5,
            },
            // DeepSeek R1: $0.55 / $2.19 per Mtok
            ProviderConfig::DeepSeek => Rates {
                input_per_tok_uc: 1,
                output_per_tok_uc: 3,
                reasoning_per_tok_uc: 3,
            },
        }
    }
}

async fn call_anthropic(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    max_output_tokens: u32,
) -> Result<(String, u64, u64, u64, u32)> {
    let body = json!({
        "model": SONNET_MODEL,
        "max_tokens": max_output_tokens + 1024, // thinking + visible
        "thinking": { "type": "enabled", "budget_tokens": 1024 },
        "system": REASONING_WORKLOAD_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    });

    let mut retries = 0u32;
    for (attempt, backoff) in std::iter::once(0).chain(OVERLOAD_BACKOFF_S.iter().copied()).enumerate() {
        if backoff > 0 {
            sleep(Duration::from_secs(backoff)).await;
        }
        let resp = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("anthropic POST")?;

        if resp.status().as_u16() == 529 && attempt < OVERLOAD_BACKOFF_S.len() {
            retries += 1;
            continue;
        }
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(anyhow!("anthropic non-2xx: {}", txt));
        }
        let parsed: AnthropicResponse = resp.json().await.context("anthropic decode")?;
        let visible = parsed
            .content
            .iter()
            .filter_map(|b| if b.ty == "text" { b.text.clone() } else { None })
            .collect::<Vec<_>>()
            .join("");
        // Anthropic does not expose thinking-token count separately;
        // we conservatively assume the thinking.budget_tokens was used.
        // (Anthropic returns thinking blocks in content but billing
        // is reported as output_tokens including thinking.)
        let thinking_tokens = 0; // billed inside output_tokens
        return Ok((
            visible,
            parsed.usage.input_tokens,
            parsed.usage.output_tokens,
            thinking_tokens,
            retries,
        ));
    }
    Err(anyhow!("exhausted retries"))
}

async fn call_openai_oseries(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    _max_output_tokens: u32,
) -> Result<(String, u64, u64, u64, u32)> {
    let body = json!({
        "model": O4_MINI_MODEL,
        "messages": [
            {"role": "system", "content": REASONING_WORKLOAD_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "reasoning_effort": "medium",
    });

    let mut retries = 0u32;
    for (attempt, backoff) in std::iter::once(0).chain(OVERLOAD_BACKOFF_S.iter().copied()).enumerate() {
        if backoff > 0 {
            sleep(Duration::from_secs(backoff)).await;
        }
        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("openai POST")?;

        if resp.status().as_u16() == 429 && attempt < OVERLOAD_BACKOFF_S.len() {
            retries += 1;
            continue;
        }
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(anyhow!("openai non-2xx: {}", txt));
        }
        let parsed: OpenAIResponse = resp.json().await.context("openai decode")?;
        let visible = parsed
            .choices
            .into_iter()
            .filter_map(|c| c.message.content)
            .collect::<Vec<_>>()
            .join("\n");
        let reasoning_tokens = parsed
            .usage
            .completion_tokens_details
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0);
        return Ok((
            visible,
            parsed.usage.prompt_tokens,
            parsed.usage.completion_tokens,
            reasoning_tokens,
            retries,
        ));
    }
    Err(anyhow!("exhausted retries"))
}

async fn call_deepseek(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    max_output_tokens: u32,
) -> Result<(String, u64, u64, u64, u32)> {
    let body = json!({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": REASONING_WORKLOAD_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output_tokens + 2000,
    });

    let mut retries = 0u32;
    for (attempt, backoff) in std::iter::once(0).chain(OVERLOAD_BACKOFF_S.iter().copied()).enumerate() {
        if backoff > 0 {
            sleep(Duration::from_secs(backoff)).await;
        }
        let resp = client
            .post("https://api.deepseek.com/v1/chat/completions")
            .header("authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("deepseek POST")?;

        if resp.status().as_u16() == 429 && attempt < OVERLOAD_BACKOFF_S.len() {
            retries += 1;
            continue;
        }
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(anyhow!("deepseek non-2xx: {}", txt));
        }
        let parsed: OpenAIResponse = resp.json().await.context("deepseek decode")?;
        let visible = parsed
            .choices
            .into_iter()
            .filter_map(|c| c.message.content)
            .collect::<Vec<_>>()
            .join("\n");
        // DeepSeek R1 separates reasoning_content from content; reasoning
        // tokens are billed inside completion_tokens.
        let reasoning_tokens = parsed
            .usage
            .completion_tokens_details
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0);
        return Ok((
            visible,
            parsed.usage.prompt_tokens,
            parsed.usage.completion_tokens,
            reasoning_tokens,
            retries,
        ));
    }
    Err(anyhow!("exhausted retries"))
}

async fn run_trial(
    provider: &ProviderConfig,
    client: &reqwest::Client,
    trial_id: usize,
) -> Result<TrialResult> {
    let mint = BudgetMint::take_authority();
    let mut budget: Budget<300_000> = Budget::mint(&mint, CAP_UC)
        .map_err(|e| anyhow!("budget mint failed: {:?}", e))?;

    let api_key = match provider {
        ProviderConfig::Anthropic => env::var("ANTHROPIC_API_KEY")?,
        ProviderConfig::OpenAI => env::var("OPENAI_API_KEY")?,
        ProviderConfig::DeepSeek => env::var("DEEPSEEK_API_KEY")?,
    };

    let rates = Rates::for_provider(provider);
    let reasoning = provider.reasoning_provider();

    let mut total_in_tok = 0u64;
    let mut total_out_tok = 0u64;
    let mut total_reasoning_tok = 0u64;
    let mut total_billed_uc = 0u64;
    let mut total_reserved_uc = 0u64;
    let mut steps = 0usize;
    let mut refused_at: i32 = -1;
    let mut refused_reason = String::new();
    let mut retries_total = 0u32;

    let mut prompt = REASONING_WORKLOAD_USER.to_string();

    for step in 0..MAX_STEPS_PER_TRIAL {
        // Pre-flight: visible_estimate = bytelen * 2.0 (Anthropic style)
        let visible_uc = (prompt.len() as u64) * 2 * rates.input_per_tok_uc
            + (MAX_OUTPUT_TOKENS as u64) * rates.output_per_tok_uc;

        match budget.spend_with_reasoning(visible_uc, reasoning) {
            Ok((new_budget, _receipt)) => {
                budget = new_budget;
                total_reserved_uc +=
                    visible_uc + reasoning.reasoning_reservation();
            }
            Err(e) => {
                refused_at = step as i32;
                refused_reason = format!("{:?}", e);
                break;
            }
        }

        let call_result = match provider {
            ProviderConfig::Anthropic => {
                call_anthropic(client, &api_key, &prompt, MAX_OUTPUT_TOKENS).await
            }
            ProviderConfig::OpenAI => {
                call_openai_oseries(client, &api_key, &prompt, MAX_OUTPUT_TOKENS).await
            }
            ProviderConfig::DeepSeek => {
                call_deepseek(client, &api_key, &prompt, MAX_OUTPUT_TOKENS).await
            }
        };

        match call_result {
            Ok((response_text, in_tok, out_tok, reasoning_tok, retries)) => {
                retries_total += retries;
                total_in_tok += in_tok;
                total_out_tok += out_tok;
                total_reasoning_tok += reasoning_tok;

                // Compute billed (subtract reasoning since out_tok may
                // include it on some providers; we account separately
                // using the provider's reported reasoning_tokens field).
                let visible_out_tok = out_tok.saturating_sub(reasoning_tok);
                let in_uc = in_tok * rates.input_per_tok_uc;
                let out_uc = visible_out_tok * rates.output_per_tok_uc;
                let reasoning_uc = reasoning_tok * rates.reasoning_per_tok_uc;
                let call_uc = in_uc + out_uc + reasoning_uc;
                total_billed_uc += call_uc;
                steps += 1;

                // Stop if the response is short (heuristic for completion)
                if response_text.len() < 200 && step > 0 {
                    break;
                }

                // Continue the loop with a follow-up
                prompt = format!(
                    "Verify your previous answer by recomputing the meeting time and position from scratch. \
                     Previous answer: {}",
                    response_text.chars().take(500).collect::<String>()
                );
            }
            Err(e) => {
                refused_at = step as i32;
                refused_reason = format!("api_error: {}", e);
                break;
            }
        }
    }

    let overshoot = if total_billed_uc > CAP_UC { 1 } else { 0 };

    Ok(TrialResult {
        trial_id,
        provider_label: provider.label().to_string(),
        steps,
        total_input_tokens: total_in_tok,
        total_output_tokens: total_out_tok,
        total_reasoning_tokens: total_reasoning_tok,
        total_billed_uc,
        total_reserved_uc,
        overshoot,
        refused_at_step: refused_at,
        refused_reason,
        retries_total,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut provider_str = "anthropic".to_string();
    let mut n_trials: usize = 20;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" => {
                provider_str = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--provider requires a value"))?
                    .clone();
                i += 2;
            }
            "--n" => {
                n_trials = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--n requires a value"))?
                    .parse()?;
                i += 2;
            }
            _ => i += 1,
        }
    }

    let provider = ProviderConfig::from_str(&provider_str)?;
    let out_dir = PathBuf::from("multiway/sweep_results");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(provider.output_filename());
    let file = File::create(&out_path)?;
    let mut writer = csv::Writer::from_writer(file);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    println!("=== {} N={} ===", provider.label(), n_trials);
    println!("    -> {:?}", out_path);

    let mut overshoots = 0;
    for trial in 0..n_trials {
        print!("    trial {}/{} ", trial + 1, n_trials);
        let row = run_trial(&provider, &client, trial).await?;
        let indicator = if row.overshoot == 1 { "OVER" } else { "ok" };
        let refused_msg = if row.refused_at_step >= 0 {
            format!(" refused@{}", row.refused_at_step)
        } else {
            String::new()
        };
        println!(
            "steps={} billed={}uc reasoning_tok={} [{}]{}",
            row.steps, row.total_billed_uc, row.total_reasoning_tokens,
            indicator, refused_msg
        );
        if row.overshoot == 1 {
            overshoots += 1;
        }
        writer.serialize(&row)?;
        writer.flush()?;
    }

    println!();
    println!("SUMMARY: {}/{} dollar-overshoot", overshoots, n_trials);
    println!("         CSV: {:?}", out_path);
    Ok(())
}