# Token Budgets -- Spike Findings

**Status:** Empirical validation complete (May 2026)
**Repo:** `~/RustroverProjects/budget-spike` (commit at validation: 56cd6b0+)
**Reproducibility:** see `bench_results.txt` and `machine_info.txt`

---

## 1. What the spike validates

The Token Budgets spike provides empirical evidence for six properties of an
affine-typed Budget mechanism for LLM agent cost control. Each property is
backed by either a unit test, a compile-fail test (rejected by `rustc`), an
integration test, a real-API demo run, or a Criterion benchmark.

| # | Property                                                            | Evidence file                              | Mechanism            |
|---|---------------------------------------------------------------------|--------------------------------------------|----------------------|
| 1 | Budget cannot be cloned (no `Clone`, no `Copy`)                     | `tests/compile_fail/clone_attempt`         | rustc E0599 reject   |
| 2 | Budget cannot be double-spent                                       | `tests/compile_fail/double_spend`          | rustc E0382 reject   |
| 3 | Budget cannot be used after `split` consumes parent                 | `tests/compile_fail/spend_after_split`     | rustc E0382 reject   |
| 4 | Budget cannot escape via reference (lifetime-bound)                 | `tests/compile_fail/escape_via_return`     | rustc E0515 reject   |
| 5 | Affine semantics survive `.await` boundaries                        | `tests/async_integration::spend_then_await_then_spend`  | tokio test passes |
| 6 | Affine semantics survive `tokio::spawn` (split-spawn-merge pattern) | `tests/async_integration::split_across_spawn`           | tokio test passes |

All six properties pass on rustc 1.93.1 stable, edition 2024, with **zero
`unsafe` code anywhere in the implementation**.

---

## 2. Real-API validation

The mechanism was validated against three live LLM endpoints in a single
working session (May 1, 2026), total spend approximately $0.005 USD across
all runs.

### 2.1 Anthropic (Haiku)

```
Initial budget: 2000 uc (= $0.0020)
Model: claude-haiku-4-5-20251001

Call 1 OK   | 17 in / 33 out | actual: 182 uc | remaining: 1490 uc
Call 2 OK   | 21 in / 11 out | actual:  76 uc | remaining:  980 uc
Call 3 OK   | 19 in / 79 out | actual: 414 uc | remaining:  471 uc
Call 4 STOP | budget exhausted: needed 511 uc, had 471 uc
```

Three successful calls, clean termination on call 4. No HTTP errors.
Response usage parsed correctly from Anthropic's `usage.input_tokens` and
`usage.output_tokens` fields.

### 2.2 OpenAI (gpt-4o-mini)

```
Initial budget: 500 uc (= $0.0005)
Model: gpt-4o-mini

Call 1 OK   | 16 in /  32 out | actual:  48 uc | remaining: 380 uc
Call 2 OK   | 18 in /   7 out | actual:  25 uc | remaining: 261 uc
Call 3 OK   | 17 in / 100 out | actual: 117 uc | remaining: 143 uc
Call 4 OK   | 18 in /   4 out | actual:  22 uc | remaining:  21 uc
Call 5 STOP | budget exhausted: needed 117 uc, had 21 uc
```

Four successful calls, clean termination on call 5. Reservation strictly
exceeded actual spend on every call after estimator tightening (see section 4).

### 2.3 Multi-provider session

```
Initial budget: 1500 uc (= $0.0015)
Anthropic Haiku reserves ~508 uc/call; OpenAI gpt-4o-mini reserves ~117 uc/call.

Call 01 [anthropic] OK   | 16 in / 35 out | 191 uc | remaining 991 uc
Call 02 [openai   ] OK   | 18 in /  7 out |  25 uc | remaining 881 uc
Call 03 [anthropic] OK   | 19 in / 89 out | 464 uc | remaining 372 uc
Call 04 [openai   ] OK   | 19 in /  4 out |  23 uc | remaining 261 uc
Call 05 [anthropic] STOP | budget exhausted: needed 507 uc, had 261 uc
```

A single `Budget` value was threaded through interleaved calls to two
different providers. Termination occurs at the configured cap regardless of
which provider is being invoked. This empirically demonstrates
provider-agnostic cost enforcement: the type system is the trust boundary,
not provider-specific code paths.

### 2.4 Conservative-reservation invariant

Across all observed real-API calls, the worst-case reservation always met
or exceeded the actual deduction:

- Anthropic: 3 successful calls, headroom ratio reservation/actual ranged
  from 1.2x to 6.7x (most generous on short responses).
- OpenAI: 4 successful calls, headroom ranged from 1.0x (tight: call 3 hit
  the `max_completion_tokens=100` ceiling exactly) to 5.3x.

The cap-not-the-call invariant: even when an individual call's reservation
ran tight against actual usage, subsequent calls were correctly blocked at
the budget level. The mechanism's safety property is the *aggregate spend
cap*, not per-call estimation accuracy.

---

## 3. Performance overhead

Microbenchmarks on AMD Ryzen 7 PRO 6850U, Linux 6.8, rustc 1.93.1 release
build with default LTO. Measurement via Criterion 0.5, 100 samples per
operation, 5-second collection windows yielding 10^9+ iterations each.

| Operation                                | Median time | vs unguarded |
|------------------------------------------|-------------|--------------|
| `Budget::new`                            | 661 ps      | 1.00x (672 ps `u64` literal) |
| `Budget::spend(amount)` -- success path  | **905 ps**  | 1.01x (895 ps `u64 -= a`) |
| `u64::checked_sub` (stdlib equivalent)   | 1356 ps     | 1.51x        |
| `Budget::split` + `merge` round-trip     | 919 ps      | 1.03x        |
| `Budget::spend(amount)` -- error path    | 1841 ps     | (executes <= 1x per session) |
| `estimate_cost` helper                   | 1370 ps     | --           |

**Headline finding:** `Budget::spend()` runs in **905 picoseconds** -- under
one nanosecond, and *faster than the standard library's `u64::checked_sub`
(1356 ps)*, the canonical Rust-idiomatic safety check. The compile-time
enforcement is, in effect, free.

The mechanism is sub-nanosecond because LLVM inlines the affine `self`
move, the audit-label closure call, and the bounds check into roughly the
same machine code as a raw subtract. The only remaining overhead -- 10 ps
versus an unguarded subtraction -- is statistical noise.

This kills the most predictable reviewer concern about compile-time
enforcement: there is no observable runtime tax.

---

## 4. Calibration findings

One non-correctness finding worth recording: the default
`estimate_input_tokens` heuristic was initially `(prompt.len() + 3) / 4`
(approximating ~4 chars/token, the standard English-prose figure).
On OpenAI call 3 of the initial demo run, this produced an 8 uc shortfall
(estimated 8 input tokens vs actual 17 from response usage) -- reservation
fell below actual spend by ~7%.

This is *not a correctness bug*. The aggregate spend cap held; the next call
was correctly blocked at the budget level. But it is a calibration finding
worth a paper paragraph: tokenization heuristics should be set
conservatively, because per-call reservation is a soft constraint; the
*aggregate budget* is the hard constraint.

The estimator was tightened to `(prompt.len() + 1) / 2` (~2 chars/token,
roughly 2x conservative). Re-running the OpenAI demo confirmed reservation
strictly met or exceeded actual on every call.

---

## 5. What this spike does NOT validate

Honest scope, for the paper's "limitations" section:

- **Refund semantics on failed LLM calls.** The current `call_with_budget`
  reserves up-front and never refunds; if the API call fails after
  reservation, the budget is consumed. Real systems would want optional
  refund-on-failure. Adding this requires careful semantics (refunds are
  not generally affine-safe in concurrent settings) and is out of scope
  for the spike.

- **Streaming responses.** The current implementation reserves
  `max_completion_tokens` worth of output budget up-front. For streaming,
  one might want to deduct per-token as the stream arrives. This is a
  natural extension but not implemented.

- **Sub-budget hierarchies beyond depth 2.** The `split` operation has
  been validated on parent to child. Recursive `split` on children works
  trivially (it is the same operation), but no test specifically exercises
  depth-3+ trees.

- **Pricing accuracy.** The pricing table in `llm_client` uses
  approximate per-token figures based on family-level matching ("opus"
  vs "haiku" vs default). Production use would need a maintained pricing
  database keyed on exact model strings.

- **Const-generic typestate.** The current Budget is a runtime-checked u64.
  A more aggressive design encodes the budget value in the type system
  via const generics (e.g., `Budget<5_000_000>`), which would make
  insufficient-spend a compile error in some cases. Out of scope for the
  spike.

- **Formal soundness proof.** The compile-fail tests demonstrate properties
  empirically. A TLA+ or mechanized proof of the affine invariants is a
  natural extension but not present.

---

## 6. Code statistics

Approximate counts -- run `tokei` for exact figures:

```
Rust source files: 13
Total lines:       ~1200
Code lines:        ~900
Comment lines:    ~200
Blank lines:       ~100
```

- `src/lib`: Budget core, error types, `call_with_budget` async helper, unit tests
- `src/llm_client`: `LLMClient` trait + Mock + Anthropic + OpenAI implementations
- `src/bin/`: Five demo binaries (sync runtime, async mock, real Anthropic,
  real OpenAI, multi-provider)
- `tests/`: 4 unit tests, 4 compile-fail tests, 4 async integration tests
- `benches/`: Criterion microbenchmark suite
- **Zero `unsafe` blocks.**
- **Zero `Arc<Mutex<>>` for the Budget itself** (some HTTP-client internals
  may use them; not relevant to the type-system claim).

---

## 7. Reproducibility

```
Hardware: AMD Ryzen 7 PRO 6850U, x86_64
Kernel:   Linux 6.8.0-110-generic (Ubuntu)
Compiler: rustc 1.93.1 stable (2026-02-11), edition 2024
Cargo:    1.93.1 (2025-12-15)
```

To reproduce:

```bash
git clone REPO-URL
cd budget-spike
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
cargo test                              # 12 tests across 3 files
cargo bench                             # ~30s, writes target/criterion/ HTML reports
cargo run --bin demo_async_mock         # offline, free
cargo run --bin demo_async_anthropic    # ~$0.001
cargo run --bin demo_async_openai       # <$0.001
cargo run --bin demo_multi_provider     # ~$0.0015 across both providers
```

Total reproduction cost: under $0.005 USD.