use budget_spike::Budget;

fn main() {
    let b = Budget::new(1_000);
    let _b2 = b.clone();
    let _ = b;
}