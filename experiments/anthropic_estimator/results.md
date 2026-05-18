# Adversarial AnthropicEstimator Audit (v2)

Tested: AnthropicEstimator with ByteLength base x 1.05 safety_margin
Provider: Anthropic claude-haiku-4-5-20251001 (max_tokens=1, prompt_tokens recovered from usage block)

Total runs: 35
A1 holds: 25/35
A1 violations: 10

## Per-class summary

| Class | N | Min ratio | Mean ratio | Max ratio | Min margin needed | A1 holds |
|---|---|---|---|---|---|---|
| large_tool_def | 5 | 2.2255 | 2.2255 | 2.2255 | 0.4718 | 5/5 |
| long_system_prompt | 5 | 2.1181 | 2.1181 | 2.1181 | 0.4957 | 5/5 |
| multi_turn_history | 5 | 2.3743 | 2.3743 | 2.3743 | 0.4423 | 5/5 |
| multi_tool_results | 5 | 1.9634 | 1.9634 | 1.9634 | 0.5349 | 5/5 |
| cache_control | 5 | 2.5628 | 2.5628 | 2.5628 | 0.4103 | 5/5 |
| nested_tool_schema | 5 | 0.5600 | 0.5600 | 0.5600 | 1.8750 | 0/5 |
| unicode_dense_tool_desc | 5 | 0.8469 | 0.8469 | 0.8469 | 1.2399 | 0/5 |

## Margin headroom analysis

Configured margin: 1.05x

Worst-case min margin needed across all 35 runs: **1.8750x**

**The configured 1.05x margin is INSUFFICIENT** by 82.5 percentage points on the worst-case prompt class.

Recommended `AnthropicEstimator::safety_margin`: **1.8950** (worst-case + 2% headroom).

Worst-case classes:
  - nested_tool_schema: min margin needed = 1.8750x (5 runs)
  - unicode_dense_tool_desc: min margin needed = 1.2399x (5 runs)
