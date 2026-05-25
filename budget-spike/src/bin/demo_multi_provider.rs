use std::error::Error;
use budget_spike::{
    Budget, CallError, BudgetError,
    call_with_budget,
    llm_client::{AnthropicClient, OpenAIClient},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Budget Spike: Multi-Provider Demo ===");
    println!("One Budget. Two providers. Same affine semantics.\n");

    let mut budget = Budget::new(1_500);

    let anthropic = AnthropicClient::from_env("claude-haiku-4-5-20251001")?;
    let openai = OpenAIClient::from_env("gpt-4o-mini")?;

    println!(
        "Initial budget: {} uc (= ${:.4})",
        budget.available(),
        budget.available() as f64 / 1_000_000.0
    );
    println!("Anthropic Haiku reserves ~508 uc/call; OpenAI gpt-4o-mini reserves ~108 uc/call.");
    println!("Expect termination around call 5-6.\n");

    let calls: &[(&str, &str)] = &[
        ("anthropic", "What is affine typing? One sentence."),
        ("openai",    "Is Rust memory-safe? Reply in 5 words."),
        ("anthropic", "Name 2 LLM agent frameworks. Brief."),
        ("openai",    "Translate 'hello world' to French. 2 words."),
        ("anthropic", "What is 2+2? One digit only."),
        ("openai",    "Sky blue? Yes or no."),
        ("anthropic", "What's the capital of France? One word."),
        ("openai",    "Pi to 3 digits."),
    ];

    for (i, (provider, prompt)) in calls.iter().enumerate() {
        let i = i + 1;
        let result = match *provider {
            "anthropic" => call_with_budget(&anthropic, budget, prompt, 100).await,
            "openai"    => call_with_budget(&openai,    budget, prompt, 100).await,
            _ => unreachable!(),
        };

        match result {
            Ok((remaining, resp)) => {
                println!(
                    "Call {i:02} [{provider:<9}] OK   | {} in / {} out | {} uc | remaining {} uc",
                    resp.input_tokens,
                    resp.output_tokens,
                    resp.actual_cost_micro_cents,
                    remaining.available()
                );
                println!("        response: {}", truncate(&resp.content, 70));
                budget = remaining;
            }
            Err(CallError::Budget(BudgetError::Insufficient { requested, available })) => {
                println!(
                    "Call {i:02} [{provider:<9}] STOP | budget exhausted: needed {requested} uc, had {available} uc"
                );
                println!("\nClean termination across providers.");
                println!(
                    "Reserved {} uc (= ${:.4}) total across both providers.",
                    1_500 - available,
                    (1_500 - available) as f64 / 1_000_000.0
                );
                return Ok(());
            }
            Err(other) => {
                eprintln!("Call {i:02} [{provider}] error: {other}");
                return Err(other.into());
            }
        }
    }

    println!("\nFinal remaining: {} uc", budget.available());
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