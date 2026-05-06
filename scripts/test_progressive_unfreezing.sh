#!/bin/bash
# Test progressive unfreezing at higher support sizes
# This script tests the hypothesis that supervised fine-tuning of basis weights
# and/or bases improves performance at high support sizes (2000+)

# Configuration
TTA_EPOCHS=7001
OUTPUT_DIR="data/tta_progressive_unfreezing"

# Support sizes to test - focus on high support sizes where progressive unfreezing should help
SUPPORT_SIZES="500 1000 2500 5000"

# Models to test - use the best performing model from your previous results
MODELS="100_25_inner_ts5000_shuffle"

# Use 5-session TTA
TTA_SESSIONS=5

echo "=================================="
echo "Progressive Unfreezing Test Suite"
echo "=================================="
echo ""
echo "This will run 4 experiments:"
echo "1. Baseline (no progressive unfreezing)"
echo "2. Unfreeze basis weights only"
echo "3. Unfreeze bases only"
echo "4. Unfreeze both (basis weights + bases)"
echo ""
echo "Support sizes: ${SUPPORT_SIZES}"
echo "Model: ${MODELS}"
echo ""
echo "=================================="
echo ""

# Experiment 1: Baseline (no progressive unfreezing)
echo "[1/4] Running baseline (no progressive unfreezing)..."
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes ${SUPPORT_SIZES} \
    --tta-epochs ${TTA_EPOCHS} \
    --max-adapt-sessions ${TTA_SESSIONS} \
    --models ${MODELS} \
    --output-dir "${OUTPUT_DIR}/baseline" \
    --no-plot-display

echo ""
echo "[1/4] Baseline complete!"
echo ""

# Experiment 2: Unfreeze basis weights only
echo "[2/4] Running with unfrozen basis weights..."
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes ${SUPPORT_SIZES} \
    --tta-epochs ${TTA_EPOCHS} \
    --max-adapt-sessions ${TTA_SESSIONS} \
    --models ${MODELS} \
    --output-dir "${OUTPUT_DIR}/unfreeze_basis_weights" \
    --no-plot-display \
    --progressive-unfreezing-threshold 0 \
    --unfreeze-basis-weights \
    --basis-weight-lr 1e-5

echo ""
echo "[2/4] Basis weights unfreezing complete!"
echo ""

# Experiment 3: Unfreeze bases only
echo "[3/4] Running with unfrozen bases..."
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes ${SUPPORT_SIZES} \
    --tta-epochs ${TTA_EPOCHS} \
    --max-adapt-sessions ${TTA_SESSIONS} \
    --models ${MODELS} \
    --output-dir "${OUTPUT_DIR}/unfreeze_bases" \
    --no-plot-display \
    --progressive-unfreezing-threshold 0 \
    --unfreeze-bases \
    --bases-lr 1e-6

echo ""
echo "[3/4] Bases unfreezing complete!"
echo ""

# Experiment 4: Unfreeze both basis weights and bases
echo "[4/4] Running with both unfrozen (basis weights + bases)..."
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes ${SUPPORT_SIZES} \
    --tta-epochs ${TTA_EPOCHS} \
    --max-adapt-sessions ${TTA_SESSIONS} \
    --models ${MODELS} \
    --output-dir "${OUTPUT_DIR}/unfreeze_both" \
    --no-plot-display \
    --progressive-unfreezing-threshold 0 \
    --unfreeze-basis-weights \
    --unfreeze-bases \
    --basis-weight-lr 1e-5 \
    --bases-lr 1e-6

echo ""
echo "[4/4] Both unfrozen complete!"
echo ""

echo "=================================="
echo "All experiments complete!"
echo "=================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To compare results, run:"
echo "python plot_tta_results.py \\"
echo "  ${OUTPUT_DIR}/baseline/tta_*.json \\"
echo "  ${OUTPUT_DIR}/unfreeze_basis_weights/tta_*.json \\"
echo "  ${OUTPUT_DIR}/unfreeze_bases/tta_*.json \\"
echo "  ${OUTPUT_DIR}/unfreeze_both/tta_*.json"
echo ""
