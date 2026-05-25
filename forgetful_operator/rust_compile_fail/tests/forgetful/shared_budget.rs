use token_budgets::{Budget, BudgetMint};

const CAP: u64 = 1_000;

async fn child(_budget: Budget<CAP>) -> u64 {
    0
}

#[tokio::main]
async fn main() {
    let budget = {
        let mint = BudgetMint::take_authority();
        Budget::<CAP>::mint(&mint, 100).unwrap()
    };

    let h1 = tokio::spawn(child(budget));

    let h2 = tokio::spawn(child(budget));

    let _ = h1.await;
    let _ = h2.await;
}