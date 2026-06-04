use anyhow::Result;
use token_budgets::{Budget, ByteLength, TokenEstimator};
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

#[cfg(feature = "tiktoken")]
use token_budgets::Tiktoken;

const BUDGET_CAP: u64 = 10_000_000_000;
type B = Budget<BUDGET_CAP>;
const IN_RATE_NC: u64 = 150;
const OUT_RATE_NC: u64 = 600;
const MAX_TOKENS: u32 = 256;

fn prompts(n: usize) -> Vec<String> {
    let templates = [
        "What is the chemical symbol for the element with atomic number {}?",
        "Compute {} factorial. Show your reasoning.",
        "Convert {} degrees Fahrenheit to Celsius. Show formula.",
        "What's the {}th prime number?",
        "If a recipe needs 3 cups flour for 4 people, how much for {}?",
        "Sum the integers from 1 to {}.",
        "What's {} mod 7?",
        "Find the GCD of {} and 96.",
        "What's the area of a circle with radius {}?",
        "Convert {} kilometers to miles.",
    ];

    (0..n).map(|i| templates[i % 10].replace("{}", &(i + 1).to_string())).collect()
}

struct Row {
    idx: usize,
    estimator: &'static str,
    reservation_nc: u64,
    actual_nc: u64,
    input_tokens: u64,
    output_tokens: u64,
    margin_ratio: f64,
}

async fn run_one(
    client: &reqwest::Client,
    api_key: &str,
    estimator: &dyn TokenEstimator,
    prompt: &str,
    idx: usize,
) -> Result<Option<Row>> {
    let body = serde_json::to_string(&json!({
        "model": "gpt-4o-mini",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }))?;

    let est_input = estimator.estimate(&body);
    let reservation = est_input * IN_RATE_NC + (MAX_TOKENS as u64) * OUT_RATE_NC;

    let budget = Budget::<BUDGET_CAP>::new(BUDGET_CAP)?;
    let (_after_reserve, receipt) = budget.spend_with_receipt(reservation)?;

    let resp = client.post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .body(body)
        .send().await;

    let resp = match resp {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            let s = r.status();
            let t = r.text().await.unwrap_or_default();
            eprintln!("idx={} HTTP {}: {}", idx, s, &t[..t.len().min(200)]);
            receipt.forfeit();
            return Ok(None);
        }

        Err(e) => {
            eprintln!("idx={} request err: {}", idx, e);
            receipt.forfeit();
            return Ok(None);
        }
    };

    let parsed: Value = resp.json().await?;

    let usage = match parsed.get("usage") {
        Some(u) => u,
        None => {
            receipt.forfeit();
            return Ok(None);
        }
    };

    let in_tok = usage["prompt_tokens"].as_u64().unwrap_or(0);

    let out_tok = usage["completion_tokens"].as_u64().unwrap_or(0);

    let actual = in_tok * IN_RATE_NC + out_tok * OUT_RATE_NC;

    if actual > reservation {
        eprintln!("⚠ A1 VIOLATION idx={} est={} actual={} (estimator={})",
                  idx, reservation, actual, estimator.name());
        receipt.forfeit();
        return Ok(None);
    }

    let _refund = receipt.confirm(actual)?;
    let margin = reservation as f64 / actual.max(1) as f64;

    Ok(Some(Row {
        idx,
        estimator: estimator.name(),
        reservation_nc: reservation,
        actual_nc: actual,
        input_tokens: in_tok,
        output_tokens: out_tok,
        margin_ratio: margin,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("OPENAI_API_KEY")?;
    let n_calls: usize = env::var("N_CALLS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);

    let client = reqwest::Client::builder().timeout(Duration::from_secs(60)).build()?;
    let prompts = prompts(n_calls);
    let mut rows: Vec<Row> = Vec::new();

    println!("Compare estimators (OpenAI gpt-4o-mini): ByteLength vs Tiktoken");
    println!("N = {} per estimator, max_tokens = {}", n_calls, MAX_TOKENS);
    println!("Tokenizer match: cl100k_base IS gpt-4o-mini's tokenizer (soundness contract holds).");

    let bl = ByteLength;
    println!("\n[1/2] Running ByteLength estimator...");
    let start = Instant::now();

    for (i, p) in prompts.iter().enumerate() {
        if let Some(r) = run_one(&client, &api_key, &bl, p, i).await? {
            rows.push(r);
        }

        if i % 10 == 9 {
            eprintln!("  ByteLength progress: {}/{}", i + 1, n_calls);
        }
    }

    println!("  ByteLength done in {:.1} min", start.elapsed().as_secs_f64() / 60.0);

    #[cfg(feature = "tiktoken")]
    {
        let tk = Tiktoken::cl100k_base()?;
        println!("\n[2/2] Running Tiktoken cl100k_base estimator...");

        let start = Instant::now();

        for (i, p) in prompts.iter().enumerate() {
            if let Some(r) = run_one(&client, &api_key, &tk, p, i).await? {
                rows.push(r);
            }

            if i % 10 == 9 {
                eprintln!("  Tiktoken progress: {}/{}", i + 1, n_calls);
            }
        }

        println!("  Tiktoken done in {:.1} min", start.elapsed().as_secs_f64() / 60.0);
    }

    #[cfg(not(feature = "tiktoken"))]
    {
        eprintln!("\n[2/2] SKIPPED: tiktoken feature not enabled.");
        eprintln!("       Rebuild with: cargo run --release --bin compare-estimators-openai --features tiktoken");
    }

    let mut by_est: std::collections::BTreeMap<&str, Vec<&Row>> = Default::default();

    for r in &rows {
        by_est.entry(r.estimator).or_default().push(r);
    }

    println!("\n Summary (OpenAI gpt-4o-mini, max_tokens={}) ", MAX_TOKENS);
    println!("{:<25} {:>5} {:>10} {:>10} {:>10} {:>10}",
             "Estimator", "N", "over-res", "min marg", "p50 marg", "p95 marg");

    for (name, rows) in &by_est {
        let n = rows.len() as f64;
        let total_r: u128 = rows.iter().map(|r| r.reservation_nc as u128).sum();
        let total_a: u128 = rows.iter().map(|r| r.actual_nc as u128).sum();
        let over_res = total_r as f64 / total_a.max(1) as f64;
        let mut margins: Vec<f64> = rows.iter().map(|r| r.margin_ratio).collect();
        margins.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = margins[((n * 0.5) as usize).min(rows.len() - 1)];
        let p95 = margins[((n * 0.95) as usize).min(rows.len() - 1)];
        let mn = margins[0];
        println!("{:<25} {:>5} {:>9.2}x {:>9.3}x {:>9.2}x {:>9.2}x",
                 name, rows.len(), over_res, mn, p50, p95);
    }

    let mut csv = File::create("compare_estimators_openai.csv")?;
    writeln!(csv, "idx,estimator,reservation_nc,actual_nc,input_tokens,output_tokens,margin_ratio")?;

    for r in &rows {
        writeln!(csv, "{},{},{},{},{},{},{:.6}",
                 r.idx, r.estimator, r.reservation_nc, r.actual_nc,
                 r.input_tokens, r.output_tokens, r.margin_ratio)?;
    }

    println!("\nWrote {} rows to compare_estimators_openai.csv", rows.len());

    Ok(())
}