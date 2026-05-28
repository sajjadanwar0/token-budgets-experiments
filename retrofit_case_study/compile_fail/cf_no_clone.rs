use retrofit_cases::Budget;
fn main() {
    let b = Budget::new(1000);
    let _dup = b.clone();   // ERROR: no method named `clone` / Clone not implemented
    let _ = b.available();
}
