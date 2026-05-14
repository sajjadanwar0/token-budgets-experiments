//! Demo: Real Anthropic API calls bounded by Budget.
//! Requires: export ANTHROPIC_API_KEY=sk-ant-...
//! Run: cargo run --bin demo_async_anthropic

use std::error::Error;
use budget_spike::{
    Budget, CallError, BudgetError,
    call_with_budget,
    llm_client::AnthropicClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Budget Spike: Real Anthropic Demo ===\n");

    let mut budget = Budget::new(2_000);
    let client = AnthropicClient::from_env("claude-haiku-4-5-20251001")?;

    println!(
        "Initial budget: {} uc (= ${:.4})",
        budget.available(),
        budget.available() as f64 / 1_000_000.0
    );
    println!("Model: claude-haiku-4-5-20251001 (Haiku: 1uc/in, 5uc/out)");
    println!("Per-call reservation: ~8 in * 1uc + 100 max-out * 5uc = ~508 uc.");
    println!("Expect termination around call 4-5.\n");

    let prompts = [
        "In one sentence, what is affine typing?",
        "Reply in 5 words: is Rust memory-safe?",
        "Name 2 LLM agent frameworks. Brief.",
        "Translate to French in 3 words: hello world.",
        "What is 2+2? Reply with one digit.",
        "Yes or no: is the sky blue?",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        let i = i + 1;
        match call_with_budget(&client, budget, prompt, 100).await {
            Ok((remaining, resp)) => {
                println!(
                    "Call {i} OK | {} in / {} out tokens | actual: {} uc | remaining: {} uc",
                    resp.input_tokens,
                    resp.output_tokens,
                    resp.actual_cost_micro_cents,
                    remaining.available()
                );
                println!("    response: {}", truncate(&resp.content, 70));
                budget = remaining;
            }
            Err(CallError::Budget(BudgetError::Insufficient { requested, available })) => {
                println!(
                    "Call {i} STOP | budget exhausted: needed {requested} uc, had {available} uc"
                );
                println!(
                    "\nFinal: reserved {} uc (= ${:.4}), {} uc remaining at stop.",
                    2_000 - available,
                    (2_000 - available) as f64 / 1_000_000.0,
                    available
                );
                return Ok(());
            }
            Err(other) => {
                eprintln!("Call {i} failed: {other}");
                return Err(other.into());
            }
        }
    }

    println!(
        "\nAll prompts succeeded within budget. Remaining: {} uc",
        budget.available()
    );
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push_str("...");
        out
    }
}