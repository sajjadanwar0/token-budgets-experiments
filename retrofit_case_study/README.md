# Retrofit Case Study (`retrofit_case_study/`)

Compile-verified evidence for the paper's §5 "Retrofit case study" subsection
(`sec:eval-retrofit`). It maps three **documented catalogue incidents** to
type-addressable mechanisms and proves with `cargo` + `trybuild`-style fixtures
that the buggy pattern **does not compile**.

| Incident | Issue | Mechanism | Retrofit | Buggy variant rejected with |
|----------|-------|-----------|----------|------------------------------|
| LANG-001 | langgraph #6731 | cumulative spend, no cap | pre-flight check + thread remainder | `E0382` (use of moved value) |
| CDXL-001 | codex #18335 | spawn slot not reclaimed | `split` / `merge` slot return | `E0382` (borrow of moved value) |
| ATGN-003 | autogen #5831 | swarm fanout, no conserved budget | `split` conservation | `E0599` (no `clone`; not `Clone`/`Copy`) |

## Layout

```
retrofit_case_study/
├── Cargo.toml
├── src/lib.rs                 # the retrofit (Budget type + 3 incident functions + 4 tests)
└── compile_fail/
    ├── cf_double_spend.rs      # LANG-001 double-spend  -> must fail E0382
    ├── cf_double_spend.stderr  # captured rustc diagnostic
    ├── cf_use_after_split.rs   # CDXL/ATGN use-after-split -> must fail E0382
    ├── cf_use_after_split.stderr
    ├── cf_no_clone.rs          # ATGN clone/forge -> must fail E0599
    └── cf_no_clone.stderr
```

## How to run

### 1. Runtime assertions (the retrofit behaves as claimed)
```bash
cd retrofit_case_study
cargo test
```
Expected: `test result: ok. 4 passed` — `lang001_caps_spend`,
`cdxl001_no_slot_leak`, `atgn003_fanout_conserves`,
`atgn003_fanout_refuses_overallocation`.

### 2. Compile-fail fixtures (the buggy variants do NOT type-check)
The fixtures are *meant to fail compilation*. Build the lib once, then confirm
each fixture is rejected with the expected error code:
```bash
cargo build
RLIB=target/debug/libretrofit_cases.rlib
for f in cf_double_spend cf_use_after_split cf_no_clone; do
  echo "== $f =="
  rustc --edition 2021 --extern retrofit_cases=$RLIB -L target/debug/deps \
        compile_fail/$f.rs -o /tmp/$f 2>&1 | grep -E '^error'
done
```
Expected: `cf_double_spend` → `error[E0382]`, `cf_use_after_split` →
`error[E0382]`, `cf_no_clone` → `error[E0599]`. (A fixture that *compiles*
is a failed test — it would mean the discipline no longer rejects the bug.)

### Optional: wire the fixtures into `cargo test` via the `trybuild` crate
This matches the main `token-budgets` repo's existing `tests/compile_fail/`
setup so CI fails if a fixture ever starts compiling. Add to `Cargo.toml`:
```toml
[dev-dependencies]
trybuild = "1.0"
```
Add `tests/compile_fail.rs`:
```rust
#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("compile_fail/cf_*.rs");
}
```
Then `cargo test` runs both the runtime assertions and the compile-fail checks,
comparing actual rustc output against the committed `.stderr` files.

## Where to put it (which repo)

Place it inside the **existing experiments repo as a sibling crate**, so it sits
next to the other harnesses `reproduce.sh` already bundles and is covered by one
DOI:

```
token-budgets-experiments/
├── agent_contracts_b2000/
├── budget-spike/
├── experiments/
├── forgetful_operator/
├── retrofit_case_study/     
│   ├── Cargo.toml
│   ├── src/lib.rs
│   └── compile_fail/
└── ...
```

Then add it to the workspace members (or leave it standalone — it has no
external deps) and reference it from `reproduce.sh` so the 20-claim audit also
runs `cargo test` here. The paper's Data Availability section and
`sec:eval-retrofit` both already point at `retrofit_case_study/`, so this path
keeps the paper and artifact consistent.

> Note: `src/lib.rs` here is a **self-contained teaching reconstruction** of the
> `Budget` API (no external deps) so the case study compiles on its own. It is
> not the production `token-budgets/src/lib.rs`; the production crate's
> `Budget<const MAX>` has the same affine ownership semantics, so the same three
> rustc rejections hold there too.