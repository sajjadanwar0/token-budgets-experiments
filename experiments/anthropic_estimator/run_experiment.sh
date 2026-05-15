#!/bin/bash
# AnthropicEstimator A1 validation — orchestrator script
# Run from this directory after setting ANTHROPIC_API_KEY

set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_ROOT="${TB_ROOT:-$HOME/RustroverProjects/token-budgets}"
mkdir -p "$EXP_DIR/results"

echo "============================================================"
echo "AnthropicEstimator A1 Validation Experiment"
echo "============================================================"
echo "Experiment dir: $EXP_DIR"
echo "Token Budgets repo: $TB_ROOT"
echo ""

# ===== Step 1: Preflight checks =====
echo "[1/4] Preflight checks..."

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set" >&2
    exit 1
fi

if [ ! -d "$TB_ROOT" ]; then
    echo "ERROR: token-budgets repo not found at $TB_ROOT" >&2
    echo "Set TB_ROOT environment variable to override" >&2
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "ERROR: cargo not found" >&2
    exit 1
fi

if ! python3 -c "import anthropic" 2>/dev/null; then
    echo "Installing anthropic Python SDK..."
    pip install anthropic --quiet
fi

echo "  ✓ ANTHROPIC_API_KEY set"
echo "  ✓ Repo found"
echo "  ✓ Python deps installed"

# ===== Step 2: Apply patch & rebuild =====
echo ""
echo "[2/4] Applying Rust patch and rebuilding..."

cd "$TB_ROOT"

# Check if patch is already applied
if grep -q "AnthropicEstimator::new()" src/estimator/default.rs 2>/dev/null; then
    echo "  (patch already applied; skipping)"
else
    if [ -f "$EXP_DIR/rust_patch.diff" ]; then
        # Try to apply; if it fails, the user needs to apply manually
        if git apply --check "$EXP_DIR/rust_patch.diff" 2>/dev/null; then
            git apply "$EXP_DIR/rust_patch.diff"
            echo "  ✓ Patch applied"
        else
            echo "  ⚠ Patch does not apply cleanly; please apply manually:"
            echo "    See $EXP_DIR/rust_patch_manual.md for instructions"
            echo "    Or skip if you have already configured AnthropicEstimator as default"
        fi
    fi
fi

echo "  Building release..."
cargo build --release --features anthropic-estimator 2>&1 | tail -3
echo "  ✓ Build complete"

# ===== Step 3: Run the experiment =====
echo ""
echo "[3/4] Running 30 experiments..."
echo "  (Expected wall-clock: 30-45 min, cost: ~\$0.50-\$1.00)"
echo ""

cd "$EXP_DIR"
python3 runner.py

# ===== Step 4: Analyze =====
echo ""
echo "[4/4] Analyzing results..."
python3 analyze.py results/runs.csv

echo ""
echo "============================================================"
echo "Experiment complete."
echo "Results:"
echo "  - results/runs.csv         (raw per-run data)"
echo "  - results/a1_validation.json (structured summary)"
echo "  - results/summary.md       (human-readable + paper-update suggestions)"
echo "============================================================"