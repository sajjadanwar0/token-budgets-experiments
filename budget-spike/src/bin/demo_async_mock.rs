use std::error::Error;
use budget_spike::{
    Budget, CallError, BudgetError,
    call_with_budget,
    llm_client::MockClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Budget Spike: Async Mock Demo ===\n");

    let mut budget = Budget::new(20_000);
    let client = MockClient::sonnet_like();

    println!(
        "Initial budget: {} uc (= ${:.4})",
        budget.available(),
        budget.available() as f64 / 1_000_000.0
    );
    println!("Each call reserves ~5 input * 3uc + 200 max-output * 15uc = ~3,015 uc.");
    println!("Expect termination around call 7.\n");

    for i in 1..=15u32 {
        let prompt = format!("Test prompt number {i:02}");
        match call_with_budget(&client, budget, &prompt, 200).await {
            Ok((remaining, resp)) => {
                println!(
                    "Call {i:02} OK   | actual: {} in / {} out = {} uc | remaining: {} uc",
                    resp.input_tokens,
                    resp.output_tokens,
                    resp.actual_cost_micro_cents,
                    remaining.available()
                );
                budget = remaining;
            }
            Err(CallError::Budget(BudgetError::Insufficient { requested, available })) => {
                println!(
                    "Call {i:02} STOP | budget exhausted: requested {requested} uc, only {available} uc available"
                );
                println!(
                    "\nClean termination. Total reserved: {} uc (= ${:.4})",
                    20_000 - available,
                    (20_000 - available) as f64 / 1_000_000.0
                );
                println!("No panic, no overrun, no leaked resources.");
                return Ok(());
            }
            Err(other) => {
                eprintln!("Unexpected error on call {i:02}: {other}");
                return Err(other.into());
            }
        }
    }
    println!("\nAll calls succeeded within budget.");
    println!("Final remaining: {} uc", budget.available());
    Ok(())
}