use budget_spike::Budget;

fn make_local_ref() -> &'static Budget {
    let b = Budget::new(1_000);
    &b // ERROR: cannot return reference to local variable `b`
}

fn main() {
    let _ = make_local_ref();
}