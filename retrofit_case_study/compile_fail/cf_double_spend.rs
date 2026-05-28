use retrofit_cases::Budget;
fn main() {
    let b = Budget::new(1000);
    let _b2 = b.spend(400).unwrap();   // consumes b
    let _b3 = b.spend(400).unwrap();   // ERROR: use of moved value `b`
}
