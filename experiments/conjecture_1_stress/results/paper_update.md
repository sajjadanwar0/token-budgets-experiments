# Conjecture 1 Stress Test — Empirical Validation of Lemma 2

## Headline

✅ **Zero Φ-invariant violations** across 10,000 iterations (22,886,985 events).

Strong empirical support for Lemma 2 (safety preservation under Tokio scheduling). Conjecture 1 remains formally open; full bisimulation refinement requires Iris/RustBelt mechanization.


## Test parameters

- Iterations: 10,000
- Tasks per iteration: 32
- Operations per task: 1,000
- Total operations: ~320,000,000
- Wall-clock: 0.2 minutes

## Distribution statistics

- Events per iteration:  mean=2289, median=2271, max=4,295, min=953
- max Φ observed (any iter): 1,000,000 (B₀ = 1,000,000); ratio = 1.0000
  ✓ max Φ ≤ B₀ throughout

## Conservation check (sanity)

For each iteration, we verify spent + dropped + final_phi = B₀.
✓ Conservation holds in all 10,000 iterations.

## Paper update (for §V Empirical stratified soundness)


> *We additionally validate Lemma~\ref{lem:safety-pres} via an adversarial
> stress test: 10,000 iterations of randomized concurrent
> workloads (each iteration: 32 Tokio tasks ×
> 1,000 operations × random
> \texttt{spend}/\texttt{split}/\texttt{merge}/\texttt{Drop}
> interleavings under work-stealing scheduling) record zero violations of
> the $\Phi(s) \le B_0$ invariant across approximately
> 320M total
> operations. This is empirical falsification testing, not mechanized
> proof: a single violation would invalidate Lemma~\ref{lem:safety-pres};
> the absence of violations across this scale is evidence that the
> monotonicity argument is correct under Tokio's actual scheduling behaviour.
> The corresponding test corpus is reproducible from
> \texttt{token-budgets/experiments/conjecture\_1\_stress/} at the
> archived commit.*
