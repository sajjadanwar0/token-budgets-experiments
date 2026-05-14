#!/usr/bin/env bash
set -euo pipefail

export CREWAI_TRACING_ENABLED=false; export CREWAI_DISABLE_TELEMETRY=true; export OTEL_SDK_DISABLED=true
exec < /dev/null

mkdir -p sweep_results_v29
cd sweep_results_v29

# Primary venv: 4 of 5 runtimes
source ../.venv/bin/activate
for prov in openai anthropic groq; do
    for wl in clarification arg_hallucination; do
        python3 ../multiway_compare.py --runs 10 --provider $prov \
            --workload $wl \
            --runtimes langgraph_only,langgraph_with_guard,crewai,token_capabilities \
            --output-csv "v29_${prov}_${wl}_no_autogen.csv"
    done
done
deactivate

# AutoGen venv
source ../.venv-autogen/bin/activate
for prov in openai anthropic groq; do
    for wl in clarification arg_hallucination; do
        python3 ../multiway_compare.py --runs 10 --provider $prov \
            --workload $wl --runtimes autogen --max-turns 4 \
            --output-csv "v29_${prov}_${wl}_autogen_only.csv"
    done
done
deactivate

# Concatenate
for prov in openai anthropic groq; do
    for wl in clarification arg_hallucination; do
        head -n 1 "v29_${prov}_${wl}_no_autogen.csv" > "v29_${prov}_${wl}.csv"
        tail -n +2 "v29_${prov}_${wl}_no_autogen.csv" >> "v29_${prov}_${wl}.csv"
        tail -n +2 "v29_${prov}_${wl}_autogen_only.csv" >> "v29_${prov}_${wl}.csv"
    done
done
ls -la v29_*.csv
