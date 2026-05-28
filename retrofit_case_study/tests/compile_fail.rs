#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("compile_fail/cf_*.rs");
}
