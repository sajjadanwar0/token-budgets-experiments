# Pooled Wilson 95% CI recomputation
Each CSV is re-analysed treating T=0 deterministic replicas as 1
effective observation per cell (the cell-level binary outcome is
'any run in cell has overshoot'). T>0 cells contribute N observations.

The three columns below are:
  - **raw**: per-run Wilson (paper's current method)
  - **per-cell**: pooled treating every cell as 1 obs
  - **hybrid**: T=0 cells as 1 obs, T>0 cells as N obs

| File | n_runs | n_cells | raw Wilson | per-cell Wilson | hybrid Wilson |
|------|-------:|--------:|------------|-----------------|---------------|
| `agent_contracts_lang001_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `claude_sonnet_lang001_n30_full.csv` | 180 | 6 | [0.979, 1.000] | [0.610, 1.000] | [0.610, 1.000] |
| `gateway_baseline_lang001_cap1000_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `gateway_baseline_lang001_cap2000_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `gateway_baseline_lang001_cap5000_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `gateway_baseline_lang001_cap500_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `gateway_baseline_lang001_cap540_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `gpt4o_arg_hallucination_n30_full.csv` | 180 | 6 | [0.269, 0.405] | [0.097, 0.700] | [0.097, 0.700] |
| `gpt4o_clarification_n30_full.csv` | 180 | 6 | [0.148, 0.264] | [0.097, 0.700] | [0.097, 0.700] |
| `gpt4o_lang001_n10.csv` | 60 | 6 | [0.227, 0.459] | [0.097, 0.700] | [0.097, 0.700] |
| `gpt4o_lang001_n10_full.csv` | 60 | 6 | [0.720, 0.907] | [0.436, 0.970] | [0.436, 0.970] |
| `gpt4o_lang001_n30_full.csv` | 180 | 6 | [0.772, 0.881] | [0.436, 0.970] | [0.436, 0.970] |
| `live_anthropic_n10_autogen_only.csv` | 10 | 1 | [0.722, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `live_anthropic_n10.csv` | 50 | 5 | [0.929, 1.000] | [0.566, 1.000] | [0.566, 1.000] |
| `live_anthropic_n10_no_autogen.csv` | 40 | 4 | [0.912, 1.000] | [0.510, 1.000] | [0.510, 1.000] |
| `live_groq_n10_autogen_only.csv` | 10 | 1 | [0.722, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `live_groq_n10.csv` | 39 | 4 | [0.910, 1.000] | [0.510, 1.000] | [0.510, 1.000] |
| `live_groq_n10_no_autogen.csv` | 29 | 3 | [0.883, 1.000] | [0.439, 1.000] | [0.439, 1.000] |
| `live_openai_n10_autogen_only.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `live_openai_n10.csv` | 50 | 5 | [0.259, 0.518] | [0.118, 0.769] | [0.118, 0.769] |
| `live_openai_n10_no_autogen.csv` | 40 | 4 | [0.329, 0.625] | [0.150, 0.850] | [0.150, 0.850] |
| `margin_sensitivity_margin1.0_cap2000_n15.csv` | 15 | 1 | [0.000, 0.204] | [0.000, 0.793] | [0.000, 0.793] |
| `margin_sensitivity_margin1.5_cap2000_n15.csv` | 15 | 1 | [0.000, 0.204] | [0.000, 0.793] | [0.000, 0.793] |
| `margin_sensitivity_margin2.0_cap2000_n15.csv` | 15 | 1 | [0.000, 0.204] | [0.000, 0.793] | [0.000, 0.793] |
| `margin_sensitivity_margin2.5_cap2000_n15.csv` | 15 | 1 | [0.000, 0.204] | [0.000, 0.793] | [0.000, 0.793] |
| `margin_sensitivity_margin3.0_cap2000_n15.csv` | 15 | 1 | [0.000, 0.204] | [0.000, 0.793] | [0.000, 0.793] |
| `mock_3way_n10.csv` | 30 | 3 | [0.488, 0.808] | [0.208, 0.939] | [0.208, 0.939] |
| `tokencap_lang001_limit10000_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `tokencap_lang001_limit2000_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `tokencap_lang001_limit5000_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `tokencap_lang001_limit540_n30.csv` | 30 | 1 | [0.886, 1.000] | [0.207, 1.000] | [0.207, 1.000] |
| `tokenizer_direct_arg-hallucination_cap2000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_arg-hallucination_cap5000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_clarification_cap2000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_clarification_cap5000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_cap1000_n30.csv` | 30 | 1 | [0.000, 0.114] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_cap2000_n30.csv` | 30 | 1 | [0.000, 0.114] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_cap5000_n30.csv` | 30 | 1 | [0.000, 0.114] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_cap500_n30.csv` | 30 | 1 | [0.000, 0.114] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_cap540_n30.csv` | 30 | 1 | [0.000, 0.114] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_T0.0_cap2000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.793] |
| `tokenizer_direct_lang001_T0.3_cap2000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.278] |
| `tokenizer_direct_lang001_T0.7_cap2000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.278] |
| `tokenizer_direct_lang001_T1.0_cap2000_n10.csv` | 10 | 1 | [0.000, 0.278] | [0.000, 0.793] | [0.000, 0.278] |
