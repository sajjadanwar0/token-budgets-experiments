use budget_spike::Budget;

fn main() {
    let b = Budget::new(1_000);
    let _b2 = b.clone(); // ERROR: no method named `clone` found for struct `Budget`
    let _ = b;
}