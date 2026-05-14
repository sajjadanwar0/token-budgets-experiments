use budget_spike::Budget;

fn main() {
    let b = Budget::new(1_000);
    let (b2, _) = b.spend(100, || ()).unwrap();
    let (_b3, _) = b.spend(100, || ()).unwrap(); // ERROR: use of moved value `b`
    let _ = b2;
}