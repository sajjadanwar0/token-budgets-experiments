#[test]
fn compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/double_spend.rs");
    t.compile_fail("tests/compile_fail/spend_after_split.rs");
    t.compile_fail("tests/compile_fail/escape_via_return.rs");
    t.compile_fail("tests/compile_fail/clone_attempt.rs");
}