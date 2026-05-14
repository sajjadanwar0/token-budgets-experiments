use budget_spike::Budget;

fn main() {
    let b = Budget::new(1_000);
    let (_remainder, _child) = b.split(400).unwrap();
    let _ = b.spend(100, || ()).unwrap(); // ERROR: use of moved value `b`
}