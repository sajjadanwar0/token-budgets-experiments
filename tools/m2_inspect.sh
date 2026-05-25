#!/usr/bin/env bash
# m2_inspect.sh — discovers the real shape of your multiway_compare.py
# so the M2 adapters and CLI commands can be made exactly correct.
#
# Run this from token-budgets-experiments/tools/ and paste the output
# back into the conversation. The script reads your local files only;
# it makes no API calls and modifies nothing.

set -euo pipefail

HARNESS="multiway_compare.py"
if [[ ! -f "$HARNESS" ]]; then
    echo "ERROR: $HARNESS not found in $(pwd)"
    echo "cd into token-budgets-experiments/tools/ and re-run."
    exit 1
fi

echo "======================================================================"
echo "1) Runtime registry — which adapters does your harness know about?"
echo "======================================================================"
grep -nE "RUNTIME_ADAPTERS|RUNTIME_REGISTRY|runtimes\s*=\s*\{|registered_runtimes|RUNTIMES\s*=" \
    "$HARNESS" | head -20 || true
echo
echo "-- Lines around the registry: --"
grep -nE -A 25 "RUNTIME_ADAPTERS\s*=|RUNTIMES\s*=\s*\{|RUNTIME_REGISTRY" "$HARNESS" | head -40 || true

echo
echo "======================================================================"
echo "2) Adapter function signature — what arguments do existing adapters take?"
echo "======================================================================"
grep -nE "^def run_" "$HARNESS" | head -10
echo
echo "-- First adapter body (first 25 lines): --"
FIRST_ADAPTER=$(grep -nE "^def run_" "$HARNESS" | head -1 | cut -d: -f1)
if [[ -n "${FIRST_ADAPTER:-}" ]]; then
    sed -n "${FIRST_ADAPTER},$((FIRST_ADAPTER + 25))p" "$HARNESS"
fi

echo
echo "======================================================================"
echo "3) Model selection — how does the script pick a model per provider?"
echo "======================================================================"
grep -nE "claude-|gpt-4o|llama-3|model\s*=|MODEL\s*=|PROVIDER_MODELS|MODELS\s*=" \
    "$HARNESS" | grep -vE "^\s*#" | head -20

echo
echo "======================================================================"
echo "4) Temperature handling — is it hardcoded or settable?"
echo "======================================================================"
grep -nE "temperature\s*=|TEMPERATURE" "$HARNESS" | head -10 || true

echo
echo "======================================================================"
echo "5) Output CSV row shape — what columns does the harness write?"
echo "======================================================================"
grep -nE "csv\.DictWriter|fieldnames\s*=|csv\.writer|writeheader|writerow" \
    "$HARNESS" | head -10
echo
echo "-- Field-name list if found: --"
grep -nE -A 5 "fieldnames\s*=" "$HARNESS" | head -15 || true

echo
echo "======================================================================"
echo "6) TB Python sim row — what does the existing tb_python_sim adapter look like?"
echo "======================================================================"
TBPY=$(grep -nE "^def.*tb_python" "$HARNESS" | head -1 | cut -d: -f1)
if [[ -n "${TBPY:-}" ]]; then
    sed -n "${TBPY},$((TBPY + 40))p" "$HARNESS"
else
    echo "(no tb_python_sim adapter found in $HARNESS — check if it lives in a separate file)"
    grep -rnE "^def.*tb_python" --include="*.py" .. | head -5 || true
fi

echo
echo "======================================================================"
echo "7) TB Rust impl row — is it called via subprocess or imported?"
echo "======================================================================"
TBRUST=$(grep -nE "^def.*tb_rust|tc_live_harness" "$HARNESS" | head -1 | cut -d: -f1)
if [[ -n "${TBRUST:-}" ]]; then
    sed -n "${TBRUST},$((TBRUST + 30))p" "$HARNESS"
else
    grep -rnE "tc_live_harness|tb_rust_impl" --include="*.py" .. | head -5 || true
fi

echo
echo "======================================================================"
echo "Done. Paste the full output above back into the conversation."
echo "======================================================================"