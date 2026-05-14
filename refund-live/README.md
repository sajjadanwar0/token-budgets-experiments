# refund-live

Live-API evaluation of the receipt/refund discipline on Anthropic's
Messages endpoint. This binary closes Reviewer 2's Overclaim 3:
"the receipt/refund mechanism is described in §IV-F but has zero
real-API evaluation."

## What it validates

The binary runs 10 real Anthropic API calls through the
`reserve → confirm → refund` cycle and checks four properties:

1. **A1 in vivo**: every call had `reservation ≥ actual` (no
   estimator under-counts).
2. **Refund arithmetic**: every refund equals
   `reservation − actual` exactly (no rounding/truncation bugs).
3. **Conservation**: final budget equals `initial − Σ actual`
   exactly. Micro-cents conserved end-to-end.
4. **Over-reservation rate**: how much the byte-length estimator
   over-reserved relative to actual cost. This is the empirical
   "utility tax" of the conservative discipline.

The cap is set to **$1.00** (`MAX = 1_000_000` micro-cents). Each
call costs roughly 100-500 uc, so total spend is ~$0.10-$0.20.

## How to run

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
cargo run --release --bin refund-live
```

Expected runtime: ~30-60 seconds (10 sequential API calls).

## Output

- **stdout**: per-call line showing reservation, actual, refund,
  token counts, and a summary block with conservation/A1/arithmetic
  validation.
- **refund_live_results.csv**: per-call records suitable for
  inclusion in the paper.

## What this evaluates that the existing TB harness does not

The main `tc_live_harness` uses the simpler `spend()` path with no
refund: it debits the reservation and never recovers the
over-reservation. This binary uses the full receipt mechanism:
`spend_with_receipt(reserved) → receipt.confirm(actual) → refund.apply_to(budget)`.

If the binary completes successfully with "✓ Conservation
verified" and "✓ Refund arithmetic" lines, that is direct
real-API evidence that the receipt/refund discipline:
- Conserves micro-cents across multiple calls
- Has correct arithmetic
- Holds A1 (under the full-request-body byte-length estimator)
- Produces a measurable over-reservation rate (typically 80-200%
  on simple Q&A prompts)

## Failure modes

- **A1 violation**: would print `✗ A1 VIOLATED` with the offending
  calls. Would also cause `receipt.confirm(actual)` to return
  `Err(ExceedsMax)` and abort the run.
- **Network failure**: prints the HTTP status + body and aborts.
- **Conservation violation**: would print `✗ CONSERVATION VIOLATED`
  with the discrepancy.

## Integration with the paper

The CSV produced by this binary should be added to the paper's
artifact alongside `a1_rerun_results.csv` and
`fair_baseline_results.csv`. Paper §IV-F should add a citation:
"the receipt/refund mechanism is validated end-to-end on the
Anthropic Messages endpoint (10 calls, see
`refund-live/refund_live_results.csv`)."
