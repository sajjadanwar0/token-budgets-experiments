use token_budgets::{Budget, BudgetMint};

const CAP: u64 = 1_000;

fn main() {
    let mint = BudgetMint::take_authority();
    let parent = Budget::<CAP>::mint(&mint, 100).unwrap();
    let (_child, _remainder) = parent.split(50).unwrap();

    let _x = parent.micro_cents();
}