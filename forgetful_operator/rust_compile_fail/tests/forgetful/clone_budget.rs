use token_budgets::{Budget, BudgetMint};

const CAP: u64 = 1_000;

fn main() {
    let mint = BudgetMint::take_authority();
    let budget = Budget::<CAP>::mint(&mint, 100).unwrap();
    let _b2 = budget.clone();
}