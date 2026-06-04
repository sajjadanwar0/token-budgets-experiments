use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use token_budgets::{Budget, BudgetMint};

const SONNET_MODEL:   &str = "claude-sonnet-4-5-20250929";
const O4_MINI_MODEL:  &str = "o4-mini";
const DEEPSEEK_MODEL: &str = "deepseek-reasoner";

const CAP_UC_LOOSE: u64 = 300_000;
const CAP_UC_TIGHT_ANTHROPIC: u64 = 100_000;
const CAP_UC_TIGHT_OPENAI:    u64 = 50_000;
const CAP_UC_TIGHT_DEEPSEEK:  u64 = 50_000;
const MAX_OUTPUT_TOKENS: u32 = 500;
const MAX_STEPS_PER_TRIAL: usize = 8;
const OVERLOAD_BACKOFF_S: &[u64] = &[2, 5, 10, 20, 30];

const DEFAULT_ANTHROPIC_REASONING_UC: u64 = 15_360;
const DEFAULT_OPENAI_REASONING_UC:    u64 = 2_000;
const DEFAULT_DEEPSEEK_REASONING_UC:  u64 = 1_200;

const REASONING_WORKLOAD_SYSTEM: &str =
    "You are a step-by-step reasoning assistant. Show your reasoning explicitly. \
     After your thinking, give a concise answer.";

fn workload_prompt(name: &str) -> Result<&'static str> {
    match name {
        "train-meeting" => Ok(
            "A train leaves Station A at 9:00 AM travelling at 80 km/h. Another train \
             leaves Station B (240 km away) at 9:30 AM travelling toward A at 100 km/h. \
             At what time and how far from Station A do they meet? Show all work."
        ),

        "integral" => Ok(
            "Compute the definite integral of (3x^2 + 2x - 5) sin(x) from 0 to pi. \
             Show every integration-by-parts step and verify by differentiation."
        ),

        "optimisation" => Ok(
            "A farmer has 800 meters of fencing and wants to enclose a rectangular field \
             along a straight river so that no fence is needed on the river side. What \
             dimensions maximise the area? Verify by computing the second derivative."
        ),

        "sequence" => Ok(
            "Find the closed form for the sequence a_n defined by a_0 = 1, a_1 = 3, \
             a_n = 4 a_{n-1} - 4 a_{n-2}. Verify by substitution into the recurrence \
             and by checking the first five terms."
        ),

        other => Err(anyhow!("unknown workload: {}. Choose one of: \
                              train-meeting, integral, optimisation, sequence", other)),
    }
}

#[derive(Debug, Serialize)]
struct TrialResult {
    trial_id: usize,
    provider_label: String,
    workload: String,
    cap_uc: u64,
    reasoning_uc_per_call: u64,
    steps: usize,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_reasoning_tokens: u64,
    total_billed_uc: u64,
    total_reserved_uc: u64,
    overshoot: u8,
    a1_violation: u8,
    refused_at_step: i32,
    refused_reason: String,
    retries_total: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse { content: Vec<AnthropicContentBlock>, usage: AnthropicUsage }
#[derive(Debug, Deserialize)]
struct AnthropicContentBlock { #[serde(rename = "type")] ty: String, text: Option<String> }
#[derive(Debug, Deserialize)]
struct AnthropicUsage { input_tokens: u64, output_tokens: u64 }

#[derive(Debug, Deserialize)]
struct OpenAIResponse { choices: Vec<OpenAIChoice>, usage: OpenAIUsage }
#[derive(Debug, Deserialize)]
struct OpenAIChoice { message: OpenAIMessage }
#[derive(Debug, Deserialize)]
struct OpenAIMessage { content: Option<String> }
#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u64, completion_tokens: u64,
    #[serde(default)] completion_tokens_details: Option<CompletionTokensDetails>,
}
#[derive(Debug, Deserialize, Default)]
struct CompletionTokensDetails { #[serde(default)] reasoning_tokens: u64 }

enum ProviderConfig { Anthropic, OpenAI, DeepSeek }
impl ProviderConfig {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "anthropic" => Ok(Self::Anthropic),
            "openai"    => Ok(Self::OpenAI),
            "deepseek"  => Ok(Self::DeepSeek),
            other       => Err(anyhow!("unknown provider: {}", other)),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic_sonnet_thinking",
            Self::OpenAI    => "openai_o4mini_medium",
            Self::DeepSeek  => "deepseek_r1",
        }
    }

    fn default_reasoning_uc(&self) -> u64 {
        match self {
            Self::Anthropic => DEFAULT_ANTHROPIC_REASONING_UC,
            Self::OpenAI    => DEFAULT_OPENAI_REASONING_UC,
            Self::DeepSeek  => DEFAULT_DEEPSEEK_REASONING_UC,
        }
    }
}

struct Rates { input_per_tok_uc: u64, output_per_tok_uc: u64, reasoning_per_tok_uc: u64 }
impl Rates {
    fn for_provider(p: &ProviderConfig) -> Self {
        match p {
            ProviderConfig::Anthropic => Rates { input_per_tok_uc: 3, output_per_tok_uc: 15, reasoning_per_tok_uc: 15 },
            ProviderConfig::OpenAI    => Rates { input_per_tok_uc: 1, output_per_tok_uc: 5,  reasoning_per_tok_uc: 5  },
            ProviderConfig::DeepSeek  => Rates { input_per_tok_uc: 1, output_per_tok_uc: 3,  reasoning_per_tok_uc: 3  },
        }
    }
}

async fn call_anthropic(client: &reqwest::Client, api_key: &str, prompt: &str, max_output_tokens: u32)
                        -> Result<(String, u64, u64, u64, u32)>
{
    let body = json!({
        "model": SONNET_MODEL,
        "max_tokens": max_output_tokens + 1024,
        "thinking": { "type": "enabled", "budget_tokens": 1024 },
        "system": REASONING_WORKLOAD_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    });

    let mut retries = 0u32;
    let attempts: Vec<u64> = std::iter::once(0u64).chain(OVERLOAD_BACKOFF_S.iter().copied()).collect();

    for (idx, backoff) in attempts.iter().enumerate() {
        if *backoff > 0 { sleep(Duration::from_secs(*backoff)).await; }
        let resp = client.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body).send().await.context("anthropic POST")?;

        if resp.status().as_u16() == 529 && idx < attempts.len() - 1 { retries += 1; continue; }

        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(anyhow!("anthropic non-2xx: {}", txt));
        }
        let parsed: AnthropicResponse = resp.json().await.context("anthropic decode")?;

        let visible = parsed.content.iter()
            .filter_map(|b| if b.ty == "text" { b.text.clone() } else { None })
            .collect::<Vec<_>>().join("");

        return Ok((visible, parsed.usage.input_tokens, parsed.usage.output_tokens, 0, retries));
    }

    Err(anyhow!("exhausted retries"))
}

async fn call_openai_oseries(client: &reqwest::Client, api_key: &str, prompt: &str, _max_output_tokens: u32)
                             -> Result<(String, u64, u64, u64, u32)>
{
    let body = json!({
        "model": O4_MINI_MODEL,
        "messages": [
            {"role": "system", "content": REASONING_WORKLOAD_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "reasoning_effort": "medium",
    });

    let mut retries = 0u32;
    let attempts: Vec<u64> = std::iter::once(0u64).chain(OVERLOAD_BACKOFF_S.iter().copied()).collect();

    for (idx, backoff) in attempts.iter().enumerate() {
        if *backoff > 0 { sleep(Duration::from_secs(*backoff)).await; }

        let resp = client.post("https://api.openai.com/v1/chat/completions")
            .header("authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body).send().await.context("openai POST")?;

        if resp.status().as_u16() == 429 && idx < attempts.len() - 1 { retries += 1; continue; }

        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(anyhow!("openai non-2xx: {}", txt));
        }

        let parsed: OpenAIResponse = resp.json().await.context("openai decode")?;

        let visible = parsed.choices.into_iter().filter_map(|c| c.message.content)
            .collect::<Vec<_>>().join("\n");

        let reasoning_tokens = parsed.usage.completion_tokens_details
            .map(|d| d.reasoning_tokens).unwrap_or(0);

        return Ok((visible, parsed.usage.prompt_tokens, parsed.usage.completion_tokens,
                   reasoning_tokens, retries));
    }

    Err(anyhow!("exhausted retries"))
}

async fn call_deepseek(client: &reqwest::Client, api_key: &str, prompt: &str, max_output_tokens: u32)
                       -> Result<(String, u64, u64, u64, u32)>
{
    let body = json!({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": REASONING_WORKLOAD_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_output_tokens + 2000,
    });

    let mut retries = 0u32;

    let attempts: Vec<u64> = std::iter::once(0u64).chain(OVERLOAD_BACKOFF_S.iter().copied()).collect();

    for (idx, backoff) in attempts.iter().enumerate() {
        if *backoff > 0 { sleep(Duration::from_secs(*backoff)).await; }
        let resp = client.post("https://api.deepseek.com/v1/chat/completions")
            .header("authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body).send().await.context("deepseek POST")?;

        if resp.status().as_u16() == 429 && idx < attempts.len() - 1 { retries += 1; continue; }

        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(anyhow!("deepseek non-2xx: {}", txt));
        }

        let parsed: OpenAIResponse = resp.json().await.context("deepseek decode")?;

        let visible = parsed.choices.into_iter().filter_map(|c| c.message.content)
            .collect::<Vec<_>>().join("\n");

        let reasoning_tokens = parsed.usage.completion_tokens_details
            .map(|d| d.reasoning_tokens).unwrap_or(0);

        return Ok((visible, parsed.usage.prompt_tokens, parsed.usage.completion_tokens,
                   reasoning_tokens, retries));
    }

    Err(anyhow!("exhausted retries"))
}

struct Config {
    provider: ProviderConfig,
    workload: String,
    n_trials: usize,
    cap_uc: u64,
    cap_mode_label: &'static str,
    reasoning_uc: u64,
}

async fn run_trial(cfg: &Config, client: &reqwest::Client, trial_id: usize) -> Result<TrialResult> {
    let mint = BudgetMint::take_authority();

    let mut budget = Budget::<CAP_UC_LOOSE>::mint(&mint, cfg.cap_uc)
        .map_err(|e| anyhow!("budget mint failed: {:?}", e))?;

    let api_key = match cfg.provider {
        ProviderConfig::Anthropic => env::var("ANTHROPIC_API_KEY")?,
        ProviderConfig::OpenAI    => env::var("OPENAI_API_KEY")?,
        ProviderConfig::DeepSeek  => env::var("DEEPSEEK_API_KEY")?,
    };

    let rates = Rates::for_provider(&cfg.provider);

    let mut total_in_tok = 0u64;
    let mut total_out_tok = 0u64;
    let mut total_reasoning_tok = 0u64;
    let mut total_billed_uc = 0u64;
    let mut total_reserved_uc = 0u64;
    let mut steps = 0usize;
    let mut refused_at: i32 = -1;
    let mut refused_reason = String::new();
    let mut retries_total = 0u32;

    let mut prompt = workload_prompt(&cfg.workload)?.to_string();

    for step in 0..MAX_STEPS_PER_TRIAL {

        let visible_uc = (prompt.len() as u64) * 2 * rates.input_per_tok_uc
            + (MAX_OUTPUT_TOKENS as u64) * rates.output_per_tok_uc;

        let per_call_reserve = visible_uc + cfg.reasoning_uc;

        match budget.spend(per_call_reserve) {
            Ok(new_budget) => {
                budget = new_budget;
                total_reserved_uc += per_call_reserve;
            }

            Err(e) => {
                refused_at = step as i32;
                refused_reason = format!("{:?}", e);
                break;
            }
        }

        let call_result = match cfg.provider {
            ProviderConfig::Anthropic => call_anthropic(client, &api_key, &prompt, MAX_OUTPUT_TOKENS).await,
            ProviderConfig::OpenAI    => call_openai_oseries(client, &api_key, &prompt, MAX_OUTPUT_TOKENS).await,
            ProviderConfig::DeepSeek  => call_deepseek(client, &api_key, &prompt, MAX_OUTPUT_TOKENS).await,
        };

        match call_result {
            Ok((response_text, in_tok, out_tok, reasoning_tok, retries)) => {
                retries_total += retries;
                total_in_tok        += in_tok;
                total_out_tok       += out_tok;
                total_reasoning_tok += reasoning_tok;

                let visible_out_tok = out_tok.saturating_sub(reasoning_tok);
                let in_uc        = in_tok          * rates.input_per_tok_uc;
                let out_uc       = visible_out_tok * rates.output_per_tok_uc;
                let reasoning_uc = reasoning_tok   * rates.reasoning_per_tok_uc;
                total_billed_uc += in_uc + out_uc + reasoning_uc;
                steps += 1;

                if response_text.len() < 200 && step > 0 { break; }
                prompt = format!(
                    "Verify your previous answer by recomputing from scratch. Previous answer: {}",
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

    let overshoot    = if total_billed_uc > cfg.cap_uc        { 1 } else { 0 };
    let a1_violation = if total_billed_uc > total_reserved_uc { 1 } else { 0 };

    Ok(TrialResult {
        trial_id,
        provider_label: cfg.provider.label().to_string(),
        workload: cfg.workload.clone(),
        cap_uc: cfg.cap_uc,
        reasoning_uc_per_call: cfg.reasoning_uc,
        steps,
        total_input_tokens: total_in_tok,
        total_output_tokens: total_out_tok,
        total_reasoning_tokens: total_reasoning_tok,
        total_billed_uc,
        total_reserved_uc,
        overshoot,
        a1_violation,
        refused_at_step: refused_at,
        refused_reason,
        retries_total,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut provider_str = "anthropic".to_string();
    let mut workload = "train-meeting".to_string();
    let mut n_trials: usize = 20;
    let mut tight_cap = false;
    let mut reasoning_uc_override: Option<u64> = None;

    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--provider"     => { provider_str = args[i+1].clone(); i += 2; }
            "--workload"     => { workload     = args[i+1].clone(); i += 2; }
            "--n"            => { n_trials     = args[i+1].parse()?; i += 2; }
            "--tight-cap"    => { tight_cap    = true; i += 1; }
            "--reasoning-uc" => { reasoning_uc_override = Some(args[i+1].parse()?); i += 2; }
            _ => i += 1,
        }
    }

    workload_prompt(&workload)?;

    let provider = ProviderConfig::from_str(&provider_str)?;
    let reasoning_uc = reasoning_uc_override.unwrap_or_else(|| provider.default_reasoning_uc());

    let (cap_uc, cap_mode_label) = if tight_cap {
        match provider {
            ProviderConfig::Anthropic => (CAP_UC_TIGHT_ANTHROPIC, "tight"),
            ProviderConfig::OpenAI    => (CAP_UC_TIGHT_OPENAI,    "tight"),
            ProviderConfig::DeepSeek  => (CAP_UC_TIGHT_DEEPSEEK,  "tight"),
        }
    } else { (CAP_UC_LOOSE, "loose") };

    let cfg = Config { provider, workload: workload.clone(), n_trials, cap_uc, cap_mode_label, reasoning_uc };

    let out_dir = PathBuf::from("../multiway/sweep_results");
    std::fs::create_dir_all(&out_dir)?;

    let out_path = out_dir.join(format!(
        "reasoning_eval_v2_{}_{}_{}_resv{}_n{}.csv",
        cfg.provider.label(),
        cfg.workload.replace('-', "_"),
        cfg.cap_mode_label, reasoning_uc, n_trials
    ));

    let file = File::create(&out_path)?;

    let mut writer = csv::Writer::from_writer(file);

    let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;

    println!("reasoning-eval");
    println!("    provider:                  {}", cfg.provider.label());
    println!("    workload:                  {}", cfg.workload);
    println!("    cap_uc:                    {} uc ({})", cfg.cap_uc, cap_mode_label);
    println!("    reasoning_uc per call:     {}", cfg.reasoning_uc);
    println!("    n_trials:                  {}", n_trials);
    println!("    output:                    {:?}", out_path);
    println!();

    let mut overshoots = 0;
    let mut a1_violations = 0;

    for trial in 0..n_trials {
        print!("    trial {}/{} ", trial + 1, n_trials);
        let row = run_trial(&cfg, &client, trial).await?;
        let osh = if row.overshoot == 1 { "OVER" } else { "ok" };
        let a1  = if row.a1_violation == 1 { "A1-FAIL" } else { "A1-OK" };
        let refused_msg = if row.refused_at_step >= 0 {
            format!(" refused@{}", row.refused_at_step)
        } else { String::new() };

        println!("steps={} billed={} reserved={} [{}] [{}]{}",
                 row.steps, row.total_billed_uc, row.total_reserved_uc,
                 osh, a1, refused_msg);

        if row.overshoot == 1     { overshoots    += 1; }

        if row.a1_violation == 1  { a1_violations += 1; }

        writer.serialize(&row)?;

        writer.flush()?;
    }

    println!();
    println!("SUMMARY:");
    println!("    dollar overshoots:    {}/{}", overshoots, n_trials);
    println!("    A1 violations:        {}/{}", a1_violations, n_trials);
    println!("    CSV:                  {:?}", out_path);

    Ok(())
}