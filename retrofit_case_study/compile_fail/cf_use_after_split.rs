use retrofit_cases::Budget;
fn main() {
    let parent = Budget::new(1000);
    let (_rem, _child) = parent.split(300).unwrap();  // consumes parent
    let _ = parent.available();                        // ERROR: borrow after move
}
