#!/bin/bash
# Train ablation variants mirroring fold0 of random_folds_20260101_211200.
# Each variant changes exactly one component vs. the baseline.
#
# Usage: bash scripts/train_ablations.sh [output_base_dir]
#
# Automatically detects free GPUs before each ablation — uses both if available,
# one if the other is busy with an external process.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${1:-ablations_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

echo "Ablation training started at ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_BASE}"

NUM_ABLATIONS=${#ABLATION_NAMES[@]}
SECS_PER_ABLATION=12850  # based on random_folds_20260101_211200 timing (~3.57h each)
if [ -n "${SINGLE_GPU}" ]; then
    TOTAL_SECS=$(( NUM_ABLATIONS * SECS_PER_ABLATION ))
    PARALLELISM="sequential (1 GPU)"
else
    TOTAL_SECS=$(( ((NUM_ABLATIONS + 1) / 2) * SECS_PER_ABLATION ))
    PARALLELISM="2-at-a-time (2 GPUs)"
fi
TOTAL_HRS=$(echo "scale=1; ${TOTAL_SECS} / 3600" | bc)
ETA=$(date -d "+${TOTAL_SECS} seconds" "+%a %b %d %H:%M")
echo "Estimated training time: ~${TOTAL_HRS}h (${NUM_ABLATIONS} ablations, ${PARALLELISM})"
echo "Estimated completion: ${ETA}"
echo ""

# -----------------------------------------------------------------------
# Fold0 sessions (seed=0, from random_folds_20260101_211200/fold0/hisi.torch)
# -----------------------------------------------------------------------
SESSIONS="MonkeyG_20150915_Session2_S1,MonkeyG_20150915_Session3_S1,MonkeyG_20150915_Session4_S1,MonkeyG_20150916_Session4_S1,MonkeyG_20150917_Session1_S1,MonkeyG_20150917_Session2_M1,MonkeyG_20150917_Session3_M1,MonkeyG_20150921_Session5_S1,MonkeyG_20150922_Session2_S1,MonkeyG_20150922_Session3_S1,MonkeyJ_20160426_Session3_S1,MonkeyJ_20160428_Session2_S1,MonkeyJ_20160428_Session3_S1,MonkeyJ_20160429_Session3_S1,MonkeyJ_20160624_Session3_S1,MonkeyJ_20160624_Session4_S1,MonkeyJ_20160625_Session4_S1,MonkeyJ_20160625_Session5_S1,MonkeyJ_20160627_Session2_S1,MonkeyJ_20160630_Session1_S1"

# -----------------------------------------------------------------------
# Shared hyperparams (mirror fold0 exactly)
# -----------------------------------------------------------------------
NUM_BASES=100
NUM_SESSIONS=20
LATENT_DIM=96
BASIS_RESIDUAL_RANK=16
TRAIN_SIZE=5000
BATCH_SIZE=500
# coadapt=false, shuffle=true, basis_residual_rank=16 → positional args below

BASE_CMD="python -u tma_standalone.py \
    ${NUM_BASES} ${NUM_SESSIONS} GPU_PLACEHOLDER false ${BASIS_RESIDUAL_RANK} ${TRAIN_SIZE} true \
    --latent-dim ${LATENT_DIM} \
    --batch-size-per-session ${BATCH_SIZE} \
    --held-in-sessions ${SESSIONS}"

# -----------------------------------------------------------------------
# Ablation definitions: (name, extra_flags)
# -----------------------------------------------------------------------
declare -a ABLATION_NAMES=(
    "baseline"
    "zscore_norm"
    "no_tanh"
    "no_ae_recon"
    "no_fro"
    "with_ortho"
    "no_rest"
    "no_l2"
    "no_tanh_no_rownorm"
)

declare -A ABLATION_FLAGS=(
    ["baseline"]=""
    ["zscore_norm"]="--normalizer zscore"
    ["no_tanh"]="--no-tanh-basis-weights"
    ["no_ae_recon"]="--lambda-ae-recon 0"
    ["no_fro"]="--lambda-fro 0"
    ["with_ortho"]="--lambda-ortho 0.05"
    ["no_rest"]="--no-rest-embeddings"
    ["no_l2"]="--lambda-l2 0"
    ["no_tanh_no_rownorm"]="--no-tanh-basis-weights --no-row-norm"
)

# -----------------------------------------------------------------------
# Run ablations using whichever GPUs are free (dynamic detection)
# -----------------------------------------------------------------------
declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""
declare -A GPU_ABLATION
declare -A GPU_START

wait_for_gpu() {
    local gpu=$1
    if [ -n "${GPU_PIDS[$gpu]}" ]; then
        local abl=${GPU_ABLATION[$gpu]}
        echo "GPU ${gpu}: waiting for ${abl}..."
        wait ${GPU_PIDS[$gpu]}
        local ec=$?
        if [ $ec -ne 0 ]; then
            echo "ERROR: ${abl} failed on GPU ${gpu} (exit code ${ec})"
            exit 1
        fi
        local dur=$(( $(date +%s) - ${GPU_START[$gpu]} ))
        echo "GPU ${gpu}: ${abl} done in ${dur}s"
        GPU_PIDS[$gpu]=""
    fi
}

# Returns 0 (true) if GPU has a Python process that isn't one of ours
gpu_externally_busy() {
    local gpu=$1
    local our0="${GPU_PIDS[0]:-NONE}"
    local our1="${GPU_PIDS[1]:-NONE}"
    nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | grep -v "^$" \
        | while read pid; do
            [ "$pid" = "$our0" ] || [ "$pid" = "$our1" ] && continue
            ps -p "$pid" -o comm= 2>/dev/null | grep -q "python" && echo "busy"
        done | grep -q "busy"
}

# Waits until a GPU is free (both from our jobs and external Python), sets PICKED_GPU
PICKED_GPU=""
pick_free_gpu() {
    while true; do
        for gpu in 0 1; do
            # Reap finished jobs so GPU_PIDS doesn't hold stale PIDs
            if [ -n "${GPU_PIDS[$gpu]}" ] && ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                wait_for_gpu $gpu
            fi
            if [ -z "${GPU_PIDS[$gpu]}" ] && ! gpu_externally_busy $gpu; then
                PICKED_GPU=$gpu
                return
            fi
        done
        sleep 30
    done
}

for abl in "${ABLATION_NAMES[@]}"; do
    # Skip if already completed (model.torch = old format, tbfm.torch = new split format)
    if [ -f "${OUTPUT_BASE}/${abl}/model.torch" ] || [ -f "${OUTPUT_BASE}/${abl}/tbfm.torch" ]; then
        echo "Skipping ${abl} (already complete)"
        continue
    fi

    pick_free_gpu
    GPU_ID=$PICKED_GPU

    OUT_DIR="${OUTPUT_BASE}/${abl}"
    FLAGS="${ABLATION_FLAGS[$abl]}"

    CMD=$(echo "${BASE_CMD}" | sed "s/GPU_PLACEHOLDER/${GPU_ID}/")
    CMD="${CMD} --out-dir ${OUT_DIR} ${FLAGS}"

    LOG="${OUTPUT_BASE}/${abl}.log"
    echo "Starting ${abl} on GPU ${GPU_ID} (log: ${LOG})"
    eval "${CMD}" > "${LOG}" 2>&1 &
    GPU_PIDS[$GPU_ID]=$!
    GPU_ABLATION[$GPU_ID]=$abl
    GPU_START[$GPU_ID]=$(date +%s)
    echo "${abl}: started (PID ${GPU_PIDS[$GPU_ID]})"
done

# Wait for remaining jobs
wait_for_gpu 0
wait_for_gpu 1

echo ""
echo "All ablations complete. Results in: ${OUTPUT_BASE}"
echo "Next: bash scripts/tta_ablations.sh ${OUTPUT_BASE}"
