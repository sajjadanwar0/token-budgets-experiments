//! Adversarial A1 stress test: deliberately constructs prompts
//! designed to maximize tokenizer/byte-length divergence and break
//! the conservative-estimator assumption. Targets:
//!
//! 1. CJK / emoji density: multi-byte UTF-8 chars, single token each
//! 2. Repeated rare tokens: BPE may produce fewer tokens than bytes
//! 3. JSON / format strings: provider-side rewriting may add tokens
//! 4. Very long output: max_tokens ceiling
//! 5. Mixed scripts: BPE merge behavior

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const MODEL: &str = "claude-haiku-4-5-20251001";

#[derive(Serialize)] struct Req<'a> { model: &'a str, max_tokens: u32, messages: Vec<Msg<'a>> }
#[derive(Serialize)] struct Msg<'a> { role: &'a str, content: &'a str }
#[derive(Deserialize)] struct Resp { usage: Usage }
#[derive(Deserialize)] struct Usage { input_tokens: u64, output_tokens: u64 }

struct Adversarial {
    class: &'static str,
    description: &'static str,
    prompt: String,
}

fn build_adversarial_corpus() -> Vec<Adversarial> {
    let mut v = vec![];

    // Class 1: CJK density
    v.push(Adversarial {
        class: "cjk_dense",
        description: "Chinese text: 3 bytes/char in UTF-8, often 1 token/char",
        prompt: "请用中文回答以下问题：什么是机器学习？请详细说明各种算法的优缺点和适用场景。深度学习和传统机器学习有什么区别？".to_string(),
    });

    // Class 2: Emoji density
    v.push(Adversarial {
        class: "emoji_dense",
        description: "Emoji: 4 bytes/char, often 1-3 tokens each",
        prompt: "🚀🌟🎯🔥💫🌈🎨🎭🎪🎢🎡🎠🎮🎲🎴🎯🎳🎰🎱🃏🎴🀄🎴🎺🎸🎼".to_string(),
    });

    // Class 3: Repeated rare token
    v.push(Adversarial {
        class: "repeated_rare",
        description: "Repeated rare word: BPE may produce one merged token per repetition",
        prompt: "supercalifragilisticexpialidocious ".repeat(20),
    });

    // Class 4: Long-output force
    v.push(Adversarial {
        class: "long_output",
        description: "Forces output to hit max_tokens cap",
        prompt: "Write a 500-word essay about the history of computing.".to_string(),
    });

    // Class 5: Mixed scripts
    v.push(Adversarial {
        class: "mixed_scripts",
        description: "English + Arabic + Hebrew + Devanagari mixed",
        prompt: "Translate to all five scripts: Hello world. مرحبا بالعالم. שלום עולם. नमस्ते दुनिया. 你好世界".to_string(),
    });

    // Class 6: JSON injection
    v.push(Adversarial {
        class: "json_dense",
        description: "Heavily nested JSON; provider may re-tokenize",
        prompt: r#"Parse this JSON: {"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":{"j":1}}}}}}}}}}"#.to_string(),
    });

    // Class 7: High whitespace
    v.push(Adversarial {
        class: "whitespace_pad",
        description: "Heavy whitespace: tokenizer may compress; byte count high",
        prompt: format!("{}word", "  ".repeat(200)),
    });

    // Class 8: Code-heavy
    v.push(Adversarial {
        class: "code_heavy",
        description: "Indented code: distinctive tokenizer behaviour",
        prompt: "def f():\n    for i in range(100):\n        for j in range(100):\n            print(i*j)".to_string(),
    });

    // Class 9: Long prompt boundary
    v.push(Adversarial {
        class: "long_prompt",
        description: "10KB prompt: stress max-input boundary",
        prompt: "Summarise the following text: ".to_string() + &"This is a sample sentence that will be repeated many times. ".repeat(150),
    });

    // Class 10: Provider-specific tool framing
    v.push(Adversarial {
        class: "tool_format",
        description: "Looks like tool-call syntax: provider may add framing tokens",
        prompt: r#"<function_calls><invoke name="search"><parameter name="q">test</parameter></invoke></function_calls>"#.to_string(),
    });

    v
}

const PER_IN_UC: u64 = 1;
const PER_OUT_UC: u64 = 5;
const MAX_OUT: u32 = 200;

fn estimate(body_bytes: usize) -> u64 {
    body_bytes as u64 * PER_IN_UC + (MAX_OUT as u64) * PER_OUT_UC
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY must be set")?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    let corpus = build_adversarial_corpus();
    let mut results = vec![];

    println!("{:>20} {:>12} {:>10} {:>10} {:>10} {:>6}",
             "class", "bytes", "reserve_uc", "actual_uc", "margin", "viol?");
    println!("{}", "-".repeat(80));

    for adv in &corpus {
        let req = Req {
            model: MODEL,
            max_tokens: MAX_OUT,
            messages: vec![Msg { role: "user", content: &adv.prompt }],
        };
        let body = serde_json::to_string(&req)?;
        let reservation = estimate(body.len());

        let resp = client.post(ANTHROPIC_URL)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(body.clone())
            .send().await?;

        if !resp.status().is_success() {
            println!("ERR class={} status={}", adv.class, resp.status());
            continue;
        }
        let parsed: Resp = resp.json().await?;
        let actual = parsed.usage.input_tokens * PER_IN_UC
                   + parsed.usage.output_tokens * PER_OUT_UC;
        let margin = reservation as f64 / actual.max(1) as f64;
        let violated = actual > reservation;

        println!("{:>20} {:>12} {:>10} {:>10} {:>10.3} {:>6}",
                 adv.class, body.len(), reservation, actual, margin,
                 if violated { "YES" } else { "no" });
        results.push((adv.class, adv.description, body.len(), reservation, actual, margin, violated));
    }

    println!();
    let violations = results.iter().filter(|r| r.6).count();
    let min_margin = results.iter().map(|r| r.5).fold(f64::INFINITY, f64::min);
    let max_margin = results.iter().map(|r| r.5).fold(0.0_f64, f64::max);
    println!("A1 violations: {} / {}", violations, results.len());
    println!("Margin range: {:.3}x to {:.3}x", min_margin, max_margin);

    if violations > 0 {
        println!("\n⚠ ADVERSARIAL A1 VIOLATIONS DETECTED");
        println!("These prompts break the byte-length estimator.");
        println!("Production deployments must use the receipt/refund");
        println!("path with provider-specific tokenizer or accept");
        println!("operational caps below tightest observed margin.");
    } else {
        println!("\n✓ Byte-length estimator sound on all 10 adversarial classes");
        println!("  Tightest margin: {:.3}x", min_margin);
    }

    Ok(())
}
