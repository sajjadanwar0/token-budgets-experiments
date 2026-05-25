#[test]
fn forgetful_operator_compile_fail_cases() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/forgetful/shared_budget.rs");
    t.compile_fail("tests/forgetful/clone_budget.rs");
    t.compile_fail("tests/forgetful/use_after_split.rs");
}
