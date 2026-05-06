#!/bin/bash
# Quick test of progressive unfreezing (fewer epochs for fast validation)
# Use this to verify the feature works before running the full test

ADAPT_SESSION="MonkeyG_20150914_Session3_S1"
TTA_EPOCHS=1000  # Reduced for quick testing
OUTPUT_DIR="data/tta_progressive_unfreezing_quick"
SUPPORT_SIZES="2500"  # Just test one high support size
MODELS="100_25_inner_ts5000_shuffle"

echo "=================================="
echo "Quick Progressive Unfreezing Test"
echo "=================================="
echo ""
echo "This is a quick test with:"
echo "  - Only 1000 epochs (vs 7001)"
echo "  - Only support size 2500"
echo "  - Only 2 experiments (baseline vs full unfreezing)"
echo ""
echo "=================================="
echo ""



# Full unfreezing
echo "[1/2] Running with progressive unfreezing (both basis weights + bases)..."
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes ${SUPPORT_SIZES} \
    --adapt-session "${ADAPT_SESSION}" \
    --tta-epochs ${TTA_EPOCHS} \
    --models ${MODELS} \
    --output-dir "${OUTPUT_DIR}/unfreeze_both" \
    --no-plot-display \
    --progressive-unfreezing-threshold 2000 \
    --unfreeze-basis-weights \
    --unfreeze-bases \
    --basis-weight-lr 1e-5 \
    --bases-lr 1e-6

echo ""
echo "[1/2] Progressive unfreezing complete!"
echo ""

# Baseline
echo "[2/2] Running baseline (no progressive unfreezing)..."
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes ${SUPPORT_SIZES} \
    --adapt-session "${ADAPT_SESSION}" \
    --tta-epochs ${TTA_EPOCHS} \
    --models ${MODELS} \
    --output-dir "${OUTPUT_DIR}/baseline" \
    --no-plot-display

echo ""
echo "[2/2] Baseline complete!"
echo ""

echo "=================================="
echo "Quick test complete!"
echo "=================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To compare results, run:"
echo "python plot_tta_results.py \\"
echo "  ${OUTPUT_DIR}/baseline/tta_*.json \\"
echo "  ${OUTPUT_DIR}/unfreeze_both/tta_*.json"
echo ""
