#!/bin/bash
# Example script for running TTA with progressive unfreezing

# Multi-GPU: Use all 4 GPUs to process jobs in parallel
python tta_testing.py \
    --use-multi-gpu \
    --support-sizes 500 1000 2000 5000 \
    --progressive-unfreezing-threshold 2000 \
    --unfreeze-basis-weights \
    --basis-weight-lr 1e-5 \
    --tta-epochs 7001 \
    --output-dir data/tta_progressive_unfreezing_multigpu

# Multi-GPU with specific GPUs: Use only GPUs 0, 1, 2
# python tta_testing.py \
#     --use-multi-gpu \
#     --gpu-ids 0 1 2 \
#     --support-sizes 500 1000 2000 5000 \
#     --progressive-unfreezing-threshold 2000 \
#     --unfreeze-basis-weights \
#     --basis-weight-lr 1e-5 \
#     --tta-epochs 7001 \
#     --output-dir data/tta_progressive_unfreezing_3gpus

# Single-GPU: Traditional single GPU execution
# python tta_testing.py \
#     --cuda-device 0 \
#     --support-sizes 500 1000 2000 5000 \
#     --progressive-unfreezing-threshold 2000 \
#     --unfreeze-basis-weights \
#     --basis-weight-lr 1e-5 \
#     --tta-epochs 7001 \
#     --output-dir data/tta_progressive_unfreezing

# Full progressive unfreezing with multi-GPU: Enable both basis weights and bases
# python tta_testing.py \
#     --use-multi-gpu \
#     --support-sizes 1000 2500 5000 \
#     --progressive-unfreezing-threshold 2000 \
#     --unfreeze-basis-weights \
#     --unfreeze-bases \
#     --basis-weight-lr 1e-5 \
#     --bases-lr 1e-6 \
#     --tta-epochs 10000 \
#     --output-dir data/tta_full_unfreezing_multigpu

# Conservative unfreezing with multi-GPU
# python tta_testing.py \
#     --use-multi-gpu \
#     --support-sizes 100 500 1000 2500 \
#     --progressive-unfreezing-threshold 1000 \
#     --unfreeze-basis-weights \
#     --basis-weight-lr 5e-6 \
#     --tta-epochs 7001 \
#     --output-dir data/tta_conservative_unfreezing_multigpu
