#!/bin/bash
# Run TTA on all trained ablation variants.
#
# Each ablation passes its training cfg overrides through to TTA so the model
# is run end-to-end with the ablation it was trained under (no_tanh tbfm at TTA,
# zscore normalizer at TTA, etc). Without this the TTA reverts to defaults and
# confounds the ablation.
#
# Usage: bash scripts/tta_ablations.sh <ablations_dir>

set -e

ABLATIONS_DIR="${1}"
if [ -z "${ABLATIONS_DIR}" ]; then
    echo "Usage: $0 <ablations_dir>"
    exit 1
fi
ABLATIONS_DIR=$(realpath "${ABLATIONS_DIR}")

# Detect free GPUs
get_gpu_flags() {
    local gpu0_busy gpu1_busy
    gpu0_busy=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | while read pid; do
        ps -p "$pid" -o comm= 2>/dev/null | grep -q python && echo "busy"
    done | grep -c busy || true)
    gpu1_busy=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | while read pid; do
        ps -p "$pid" -o comm= 2>/dev/null | grep -q python && echo "busy"
    done | grep -c busy || true)

    if [ "${gpu0_busy}" -eq 0 ] && [ "${gpu1_busy}" -eq 0 ]; then
        echo "--use-multi-gpu --gpu-ids 0 1"
    elif [ "${gpu0_busy}" -eq 0 ]; then
        echo "--cuda-device 0"
    elif [ "${gpu1_busy}" -eq 0 ]; then
        echo "--cuda-device 1"
    else
        echo "--cuda-device 1"
    fi
}

: "${SUPPORT_SIZE:=2500}"   # honor pre-set value (e.g. SUPPORT_SIZE=5000 for cloud runs)
STANDARD_TTA_FLAGS="--unfreeze-bases --progressive-unfreezing-threshold 0 --max-adapt-sessions 20 --tta-epochs 7001"

# Per-ablation TTA cfg overrides — must mirror the training overrides in scripts/train_ablations.sh
declare -A ABLATION_TTA_FLAGS=(
    ["baseline"]=""
    ["zscore_norm"]="--normalizer zscore"
    ["no_tanh"]="--no-tanh-basis-weights"
    ["no_ae_recon"]="--lambda-ae-recon 0"
    ["with_ortho"]="--lambda-ortho 0.05"
    ["no_rest"]="--zero-rest-embeddings"
    ["no_l2"]="--lambda-l2 0"
    ["no_tanh_no_rownorm"]="--no-tanh-basis-weights --no-row-norm"
)

# ABLATION_NAMES can be pre-set in the env. Bash arrays don't survive being
# exported to a child process, so callers should use SINGLE_ABLATION="<name>"
# (a regular string env var) to scope the sweep to one ablation.
if [ -n "${SINGLE_ABLATION:-}" ]; then
    ABLATION_NAMES=("${SINGLE_ABLATION}")
elif [ -z "${ABLATION_NAMES+x}" ]; then
    ABLATION_NAMES=(
        "baseline"
        "zscore_norm"
        "no_tanh"
        "no_ae_recon"
        "with_ortho"
        "no_rest"
        "no_l2"
        "no_tanh_no_rownorm"
    )
fi

# WINNERS_ONLY=1 restricts the sweep to the 4 ablations that significantly beat
# baseline at support=500. Used to prioritize the 2500 sweep.
SKIP_NO_ADAPT_AE=
if [ -n "${WINNERS_ONLY}" ]; then
    ABLATION_NAMES=(no_ae_recon with_ortho no_rest no_l2)
    SKIP_NO_ADAPT_AE=1
    echo "WINNERS_ONLY=1: restricting sweep to ${ABLATION_NAMES[*]}"
fi

for abl in "${ABLATION_NAMES[@]}"; do
    MODEL_DIR="${ABLATIONS_DIR}/${abl}"
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "WARNING: ${MODEL_DIR} not found, skipping ${abl}"
        continue
    fi

    OUT_DIR="${ABLATIONS_DIR}/tta_${abl}_${SUPPORT_SIZE}"
    if ls "${OUT_DIR}"/tta_support_*_per_session.csv 2>/dev/null | grep -q .; then
        echo "Skipping ${abl} TTA (already complete: ${OUT_DIR})"
        continue
    fi

    # GPU_FLAGS_OVERRIDE lets callers (e.g. cloud parallel runner) skip auto-detect
    # and force a specific gpu config like "--cuda-device 0".
    GPU_FLAGS="${GPU_FLAGS_OVERRIDE:-$(get_gpu_flags)}"
    EXTRA_FLAGS="${ABLATION_TTA_FLAGS[$abl]}"
    echo "=========================================="
    echo "TTA for ablation: ${abl}  [${GPU_FLAGS}]"
    echo "  Model: ${MODEL_DIR}"
    echo "  Output: ${OUT_DIR}"
    echo "  Ablation flags: ${EXTRA_FLAGS}"
    echo "  Support: ${SUPPORT_SIZE}"
    echo "=========================================="

    python -u tta_testing.py \
        --model-paths "${abl}:${MODEL_DIR}" \
        --output-dir "${OUT_DIR}" \
        --support-sizes ${SUPPORT_SIZE} \
        ${STANDARD_TTA_FLAGS} \
        ${EXTRA_FLAGS} \
        ${GPU_FLAGS} \
        2>&1 | tee "${ABLATIONS_DIR}/tta_${abl}_${SUPPORT_SIZE}.log"
done

# no_adapt_ae: use baseline model but skip AE gradient updates at TTA time
BASELINE_DIR="${ABLATIONS_DIR}/baseline"
if [ -n "${SKIP_NO_ADAPT_AE}" ]; then
    echo "WINNERS_ONLY: skipping no_adapt_ae"
elif [ -d "${BASELINE_DIR}" ]; then
    OUT_DIR="${ABLATIONS_DIR}/tta_no_adapt_ae_${SUPPORT_SIZE}"
    if ls "${OUT_DIR}"/tta_support_*_per_session.csv 2>/dev/null | grep -q .; then
        echo "Skipping no_adapt_ae TTA (already complete)"
    else
        GPU_FLAGS="${GPU_FLAGS_OVERRIDE:-$(get_gpu_flags)}"
        echo "=========================================="
        echo "TTA for ablation: no_adapt_ae  [${GPU_FLAGS}]"
        echo "  Support: ${SUPPORT_SIZE}"
        echo "=========================================="
        python -u tta_testing.py \
            --model-paths "no_adapt_ae:${BASELINE_DIR}" \
            --output-dir "${OUT_DIR}" \
            --support-sizes ${SUPPORT_SIZE} \
            --no-adapt-ae \
            ${STANDARD_TTA_FLAGS} \
            ${GPU_FLAGS} \
            2>&1 | tee "${ABLATIONS_DIR}/tta_no_adapt_ae_${SUPPORT_SIZE}.log"
    fi
else
    echo "WARNING: baseline dir not found, skipping no_adapt_ae"
fi

echo ""
echo "All TTA runs complete."
echo "Next: python analysis/compile_ablation_table.py --ablations-dir ${ABLATIONS_DIR} --support-size ${SUPPORT_SIZE}"
