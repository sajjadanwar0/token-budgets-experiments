//! Multi-provider regime. Now supports Anthropic, OpenAI, Gemini, Ollama.
//!
//! Usage:
//!   PROVIDER=ollama MODEL=llama3.2 MAX_TOKENS=1024 N_CALLS=300 \
//!     cargo run --release --bin refund-live-multi
//!
//! For Ollama, no API key needed. Default endpoint is
//! http://localhost:11434/v1/chat/completions; override with
//! OLLAMA_URL=http://other-host:11434/v1/chat/completions
//!
//! Output CSV: refund_live_ollama_llama3_2_1024_300.csv

use anyhow::{anyhow, Context, Result};
use budget_typed_cap::Budget;
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const BUDGET_CAP: u64 = 100_000_000_000;  // $100 in nano-cents
type B = Budget<BUDGET_CAP>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    Ollama,
}

impl Provider {
    fn from_env(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "anthropic" => Ok(Self::Anthropic),
            "openai"    => Ok(Self::OpenAI),
            "gemini"    => Ok(Self::Gemini),
            "ollama"    => Ok(Self::Ollama),
            other       => Err(anyhow!("unknown PROVIDER: {}", other)),
        }
    }

    fn name(&self) -> String {
        match self {
            Self::Anthropic => "anthropic-haiku-4.5".to_string(),
            Self::OpenAI    => "openai-gpt-4o-mini".to_string(),
            Self::Gemini    => "gemini-2.0-flash".to_string(),
            Self::Ollama    => format!("ollama-{}", self.model_id().replace(':', "_")),
        }
    }

    fn model_id(&self) -> String {
        match self {
            Self::Anthropic => env::var("MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string()),
            Self::OpenAI    => env::var("MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            Self::Gemini    => env::var("MODEL").unwrap_or_else(|_| "gemini-2.0-flash".to_string()),
            Self::Ollama    => env::var("MODEL").unwrap_or_else(|_| "llama3.2".to_string()),
        }
    }

    fn endpoint(&self) -> String {
        match self {
            Self::Anthropic => "https://api.anthropic.com/v1/messages".to_string(),
            Self::OpenAI    => "https://api.openai.com/v1/chat/completions".to_string(),
            Self::Gemini    => format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
                self.model_id()
            ),
            Self::Ollama    => env::var("OLLAMA_URL").unwrap_or_else(|_| {
                "http://localhost:11434/v1/chat/completions".to_string()
            }),
        }
    }

    /// Per-token rates in nano-cents (10^-9 USD).
    /// For Ollama (local), nominal rates of 1 nc each so budget arithmetic
    /// still meaningfully bounds calls but doesn't reflect real dollar cost.
    fn rates_nc(&self) -> (u64, u64) {
        match self {
            Self::Anthropic => (1000, 5000),
            Self::OpenAI    => (150, 600),
            Self::Gemini    => (75, 300),
            Self::Ollama    => (1, 1),  // local: integer placeholder rates
        }
    }

    fn requires_auth(&self) -> bool {
        !matches!(self, Self::Ollama)
    }

    fn api_key_env_var(&self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::OpenAI    => "OPENAI_API_KEY",
            Self::Gemini    => "GEMINI_API_KEY",
            Self::Ollama    => "OLLAMA_KEY",  // unused
        }
    }

    fn build_request(&self, prompt: &str, max_tokens: u32) -> Value {
        match self {
            Self::Anthropic => json!({
                "model": self.model_id(),
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }),
            Self::OpenAI | Self::Ollama => json!({
                "model": self.model_id(),
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "stream": false,
            }),
            Self::Gemini => json!({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }),
        }
    }

    fn add_auth(&self, builder: reqwest::RequestBuilder, key: &str) -> reqwest::RequestBuilder {
        match self {
            Self::Anthropic => builder
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json"),
            Self::OpenAI => builder
                .header("Authorization", format!("Bearer {}", key))
                .header("Content-Type", "application/json"),
            Self::Gemini => builder
                .header("x-goog-api-key", key)
                .header("Content-Type", "application/json"),
            Self::Ollama => builder.header("Content-Type", "application/json"),
        }
    }

    fn parse_usage(&self, resp: &Value) -> Option<(u64, u64)> {
        match self {
            Self::Anthropic => {
                let u = resp.get("usage")?;
                Some((u.get("input_tokens")?.as_u64()?, u.get("output_tokens")?.as_u64()?))
            }
            Self::OpenAI => {
                let u = resp.get("usage")?;
                Some((u.get("prompt_tokens")?.as_u64()?, u.get("completion_tokens")?.as_u64()?))
            }
            Self::Gemini => {
                let m = resp.get("usageMetadata")?;
                Some((m.get("promptTokenCount")?.as_u64()?, m.get("candidatesTokenCount")?.as_u64()?))
            }
            Self::Ollama => {
                // Ollama on OpenAI-compat endpoint returns:
                // {"usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}}
                // OR the native ollama endpoint returns:
                // {"prompt_eval_count": N, "eval_count": M}
                let u = resp.get("usage");
                if let Some(u) = u {
                    Some((
                        u.get("prompt_tokens")?.as_u64()?,
                        u.get("completion_tokens")?.as_u64()?,
                    ))
                } else {
                    Some((
                        resp.get("prompt_eval_count")?.as_u64()?,
                        resp.get("eval_count")?.as_u64()?,
                    ))
                }
            }
        }
    }
}

// ... rest of the binary (CallRecord struct, build_prompt, main) is
// IDENTICAL to the existing refund-live-multi.rs. Copy from there.
// The only changes are in the Provider enum and its impl block above.

struct CallRecord {
    idx: usize,
    reservation_nc: u64,
    actual_nc: u64,
    refund_nc: u64,
    input_tokens: u64,
    output_tokens: u64,
    latency_ms: u128,
    margin_ratio: f64,
    output_capped: bool,
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

#[tokio::main]
async fn main() -> Result<()> {
    let provider_str = env::var("PROVIDER")
        .context("PROVIDER must be set (anthropic | openai | gemini | ollama)")?;
    let provider = Provider::from_env(&provider_str)?;
    let max_tokens: u32 = env::var("MAX_TOKENS").unwrap_or_else(|_| "1024".to_string()).parse()?;
    let n_calls: usize = env::var("N_CALLS").unwrap_or_else(|_| "300".to_string()).parse()?;

    let api_key = if provider.requires_auth() {
        env::var(provider.api_key_env_var())
            .with_context(|| format!("{} must be set", provider.api_key_env_var()))?
    } else {
        String::new()
    };

    let (in_rate_nc, out_rate_nc) = provider.rates_nc();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))  // longer for local Ollama
        .build()?;

    println!("=== Provider: {} | model: {} | max_tokens: {} | N: {} ===",
             provider.name(), provider.model_id(), max_tokens, n_calls);
    println!("Endpoint: {}", provider.endpoint());
    println!("Rates: {} nc/in_tok, {} nc/out_tok", in_rate_nc, out_rate_nc);

    let mut budget: Option<B> = Some(Budget::new(BUDGET_CAP)?);
    let mut records: Vec<CallRecord> = Vec::with_capacity(n_calls);
    let mut sum_r = 0u128;
    let mut sum_a = 0u128;
    let mut sum_f = 0u128;
    let mut violations = 0usize;
    let mut api_errors = 0usize;

    let run_start = Instant::now();

    for i in 0..n_calls {
        let prompt = build_prompt(i);
        let req_value = provider.build_request(&prompt, max_tokens);
        let body = serde_json::to_string(&req_value)?;
        let body_bytes = body.len() as u64;
        let reservation = body_bytes
            .checked_mul(in_rate_nc)
            .and_then(|v| v.checked_add((max_tokens as u64).checked_mul(out_rate_nc)?))
            .context("reservation overflow")?;

        let current = budget.take().expect("budget present");
        let (after_reserve, receipt) = match current.spend_with_receipt(reservation) {
            Ok(x) => x,
            Err(e) => {
                println!("Budget exhausted at call {}: {:?}", i, e);
                break;
            }
        };

        let mut req_builder = client.post(&provider.endpoint()).body(body);
        if provider.requires_auth() {
            req_builder = provider.add_auth(req_builder, &api_key);
        } else {
            req_builder = req_builder.header("Content-Type", "application/json");
        }

        let start = Instant::now();
        let resp = match req_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                api_errors += 1;
                eprintln!("API error at call {}: {}", i, e);
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
        };
        let latency_ms = start.elapsed().as_millis();

        if !resp.status().is_success() {
            api_errors += 1;
            eprintln!("API status {} at call {}", resp.status(), i);
            receipt.forfeit();
            budget = Some(after_reserve);
            continue;
        }

        let parsed: Value = resp.json().await?;
        let (in_tok, out_tok) = match provider.parse_usage(&parsed) {
            Some(x) => x,
            None => {
                api_errors += 1;
                eprintln!("usage parse failed at call {}: {}", i,
                    serde_json::to_string(&parsed).unwrap_or_default()
                        .chars().take(300).collect::<String>());
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
        };

        let actual = in_tok.checked_mul(in_rate_nc)
            .and_then(|v| v.checked_add(out_tok.checked_mul(out_rate_nc)?))
            .context("actual overflow")?;

        if actual > reservation {
            violations += 1;
            println!("⚠ A1 VIOLATION call {}: reserve={} actual={} in={} out={}",
                     i, reservation, actual, in_tok, out_tok);
            receipt.forfeit();
            budget = Some(after_reserve);
            continue;
        }

        let refund = receipt.confirm(actual)?;
        let refund_amount = refund.amount();
        budget = Some(refund.apply_to(after_reserve)?);

        sum_r += reservation as u128;
        sum_a += actual as u128;
        sum_f += refund_amount as u128;

        records.push(CallRecord {
            idx: i, reservation_nc: reservation, actual_nc: actual, refund_nc: refund_amount,
            input_tokens: in_tok, output_tokens: out_tok, latency_ms,
            margin_ratio: reservation as f64 / actual.max(1) as f64,
            output_capped: out_tok as u32 == max_tokens,
        });

        if i % 25 == 0 || i == n_calls - 1 {
            println!("{:>4} reserve={:>10} actual={:>10} refund={:>10} in={:>4} out={:>4} ms={:>6} x={:>6.2}",
                     i, reservation, actual, refund_amount, in_tok, out_tok, latency_ms,
                     reservation as f64 / actual.max(1) as f64);
        }
    }

    let elapsed = run_start.elapsed();
    let mut ratios: Vec<f64> = records.iter().map(|r| r.margin_ratio).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ratios.len();
    let pct = |p: f64| -> f64 { if n == 0 { 0.0 } else { ratios[((n as f64 * p) as usize).min(n - 1)] } };
    let mean = ratios.iter().sum::<f64>() / n.max(1) as f64;
    let capped = records.iter().filter(|r| r.output_capped).count();

    println!();
    println!("=== Summary: {} | mt={} | N={} ===", provider.name(), max_tokens, n_calls);
    println!("Wall time:        {:.1} min", elapsed.as_secs_f64() / 60.0);
    println!("Successful:       {} ({} capped)", records.len(), capped);
    println!("A1 violations:    {}", violations);
    println!("API errors:       {}", api_errors);
    println!("Over-reservation: {:.3}x", sum_r as f64 / sum_a.max(1) as f64);
    println!("Margin: min={:.3}x p50={:.3}x mean={:.3}x p95={:.3}x max={:.3}x",
             pct(0.0), pct(0.5), mean, pct(0.95), pct(1.0));

    let csv_path = format!("refund_live_{}_{}_{}.csv",
        provider.name().replace('-', "_").replace('.', "_"),
        max_tokens, n_calls);
    let mut csv = File::create(&csv_path)?;
    writeln!(csv, "idx,reservation_nc,actual_nc,refund_nc,input_tokens,output_tokens,latency_ms,margin_ratio,output_capped")?;
    for r in &records {
        writeln!(csv, "{},{},{},{},{},{},{},{:.6},{}",
                 r.idx, r.reservation_nc, r.actual_nc, r.refund_nc,
                 r.input_tokens, r.output_tokens, r.latency_ms, r.margin_ratio,
                 r.output_capped)?;
    }
    println!("Wrote {} rows to {}", records.len(), csv_path);
    Ok(())
}
