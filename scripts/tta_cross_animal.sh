#!/bin/bash
# Run TTA on cross-animal generalization folds.
# Train on one monkey, adapt/evaluate on the other.
#
# Usage: bash scripts/tta_cross_animal.sh <cross_animal_dir>
#
# Expects <cross_animal_dir>/fold_G2J and fold_J2G from train_cross_animal.sh.

set -e

CROSS_ANIMAL_DIR="${1}"
if [ -z "${CROSS_ANIMAL_DIR}" ]; then
    echo "Usage: $0 <cross_animal_dir>"
    exit 1
fi
CROSS_ANIMAL_DIR=$(realpath "${CROSS_ANIMAL_DIR}")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"

STANDARD_TTA_FLAGS="--unfreeze-bases --progressive-unfreezing-threshold 0 --tta-epochs 7001 --support-sizes 2500"

for FOLD in fold_G2J fold_J2G; do
    MODEL_DIR="${CROSS_ANIMAL_DIR}/${FOLD}"
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "WARNING: ${MODEL_DIR} not found, skipping"
        continue
    fi

    OUT_DIR="${CROSS_ANIMAL_DIR}/tta_${FOLD}"
    echo "=========================================="
    echo "TTA for ${FOLD}"
    echo "  Model: ${MODEL_DIR}"
    echo "  Output: ${OUT_DIR}"
    echo "=========================================="

    # Adapt on all available held-out sessions (the other animal).
    # Wrapped in retry_on_preemption.sh so spot eviction triggers a restart
    # rather than aborting the whole sweep.
    bash "${RETRY}" python -u tta_testing.py \
        --model-paths "${FOLD}:${MODEL_DIR}" \
        --output-dir "${OUT_DIR}" \
        --max-adapt-sessions 40 \
        --cuda-device 1 \
        ${STANDARD_TTA_FLAGS} \
        2>&1 | tee "${CROSS_ANIMAL_DIR}/tta_${FOLD}.log"
done

echo ""
echo "Cross-animal TTA complete. Results in: ${CROSS_ANIMAL_DIR}"
