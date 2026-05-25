use std::error::Error;
use budget_spike::{
    Budget, CallError, BudgetError,
    call_with_budget,
    llm_client::OpenAIClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Budget Spike: Real OpenAI Demo ===\n");

    let mut budget = Budget::new(500);
    let client = OpenAIClient::from_env("gpt-4o-mini")?;

    println!(
        "Initial budget: {} uc (= ${:.4})",
        budget.available(),
        budget.available() as f64 / 1_000_000.0
    );
    println!("Model: gpt-4o-mini (1uc/in, 1uc/out approximations)");
    println!("Per-call reservation: ~8 in * 1uc + 100 max-out * 1uc = ~108 uc.");
    println!("Expect termination around call 5.\n");

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
                return Ok(());
            }
            Err(other) => {
                eprintln!("Call {i} failed: {other}");
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