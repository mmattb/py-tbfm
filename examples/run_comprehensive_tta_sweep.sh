#!/bin/bash
# Comprehensive TTA sweep with multi-GPU and progressive unfreezing
# This script demonstrates the full power of both features working together

# Configuration
OUTPUT_DIR="data/tta_comprehensive_sweep_$(date +%Y%m%d_%H%M%S)"
EPOCHS=7001

echo "=========================================="
echo "Comprehensive TTA Sweep"
echo "Multi-GPU + Progressive Unfreezing"
echo "=========================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Using all 4 GPUs for parallel processing"
echo ""

# Run comprehensive sweep
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes 100 250 500 1000 2500 5000 \
    --progressive-unfreezing-threshold 2000 \
    --unfreeze-basis-weights \
    --unfreeze-bases \
    --basis-weight-lr 1e-5 \
    --bases-lr 1e-6 \
    --tta-epochs ${EPOCHS} \
    --output-dir ${OUTPUT_DIR} \
    --include-vanilla-tbfm \
    --include-fresh-tbfm

echo ""
echo "=========================================="
echo "Sweep complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

# Show quick summary
echo ""
echo "Quick summary of results:"
echo ""
tail -30 ${OUTPUT_DIR}/tta_sweep_*.log
