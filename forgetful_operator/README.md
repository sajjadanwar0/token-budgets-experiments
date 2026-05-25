# Forgetful-Operator Experiment

A minimal reproduction of the M-delegation-fanout race condition
documented in catalogue cluster `M-delegation-fanout` (11 rows). The
experiment compares three implementations of multi-child budget
enforcement and demonstrates that the Rust affine discipline makes
the racy pattern structurally impossible.

## Conditions

1. **`python_racy.py`** --- Shared mutable budget without a lock,
   post-LLM spend recording. The asyncio check-then-act race fires
   under concurrent children.
2. **`python_locked.py`** --- Shared budget with `asyncio.Lock`
   around the check-then-act, plus pre-flight reservation. The
   correct operator discipline.
3. **`rust_affine/`** --- `Budget::split` into per-child sub-budgets.
   The type system makes shared-budget concurrency impossible.

Plus a `rust_compile_fail/` crate with three trybuild test cases
demonstrating the Rust equivalents of the Python racy patterns fail
to compile.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./run_experiment.sh
```

Total cost ~$1.50, wall-clock 25-40 min.

## Outputs

```
results/
  python_racy_anthropic.csv
  python_locked_anthropic.csv
  rust_affine_anthropic.csv
  summary.csv
```
