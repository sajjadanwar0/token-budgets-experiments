# AnthropicEstimator A1 Validation Results

## Headline

**A1 holds: 30/30 (100%)** on Anthropic Haiku-4.5 tool-loop workloads with AnthropicEstimator default.

Compare against byte-length baseline: **A1 holds 0/30 (0%)** in the same workload configuration.


## Comparison Table

| Estimator | A1 holds | mean est/actual ratio | range |
|---|---|---|---|
| ByteLength (baseline) | 0/30 (0%) | 0.72–0.79 | under-bounds actual |
| AnthropicEstimator | 30/30 (100%) | 1.0000 | 1.0000–1.0000 |

## Per-Workload Breakdown

| Workload | A1 holds | mean est_ratio | min |
|---|---|---|---|
| sql_retry | 10/10 | 1.0000 | 1.0000 |
| ambig_tool | 10/10 | 1.0000 | 1.0000 |
| arg_hallucination | 10/10 | 1.0000 | 1.0000 |

## Interpretation

✅ **AnthropicEstimator satisfies A1 in all measured cells.** The 30/30 byte-length failures previously reported were caused by Anthropic's tool-call encoding using short special tokens that the byte-count cannot capture; AnthropicEstimator's tokenizer-based approach captures these correctly.

The provider-stratified default proposed in §IV-A of the paper is validated: byte-length is sound for OpenAI/Groq; AnthropicEstimator is sound for Anthropic. The paper's Lemma~\ref{lem:tight-estimator} now has direct empirical support.


## Paper Update Required

Replace the abstract's claim *'AnthropicEstimator... uses Anthropic's actual tokenizer rather than a byte-length upper bound, sidestepping the 30/30 byte-length-A1 failures'* with:

> *'AnthropicEstimator satisfies A1 on 30/30 measured runs across three tool-loop workloads (mean est/actual ratio 1.0000, range [1.0000, 1.0000]). The provider-stratified default thereby achieves empirical A1-soundness across all three live providers in our evaluation.'*
