#!/bin/bash
# Sweep script for basis residual rank (rr), MLP hidden dimension, and stim embedding dimension
# Fixed parameters:
# - Latent dim: 96
# - Batch size: 500/session × 25 sessions = 12500 total
# - Number of sessions: 25
# - Training: Use all available training data
# - Evaluation: 5-session, 500 support TTA

set -e  # Exit on error

# Configuration
NUM_BASES=100
NUM_SESSIONS=25
LATENT_DIM=96
BATCH_SIZE_PER_SESSION=500  # Batch size per session for data loading
TRAIN_SIZE=5000  # Use all training data (5000 is default full training set)
COADAPT=false
SHUFFLE=true

# GPU configuration - use both GPUs for parallel training
GPU_IDS=(0 1)

# Sweep parameters
RR_VALUES=(8 32)
MLP_HIDDEN_VALUES=(16 32)
EMBED_DIM_STIM_VALUES=(10 15 20)

# Output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="sweep_rr_mlp_${TIMESTAMP}"
MODEL_DIR="${SWEEP_DIR}/models"
TTA_RESULTS_DIR="${SWEEP_DIR}/tta_results"

mkdir -p "${MODEL_DIR}"
mkdir -p "${TTA_RESULTS_DIR}"

echo "=========================================="
echo "Sweep Configuration"
echo "=========================================="
echo "Num bases: ${NUM_BASES}"
echo "Num sessions: ${NUM_SESSIONS}"
echo "Latent dim: ${LATENT_DIM}"
echo "Batch size per session: ${BATCH_SIZE_PER_SESSION}"
echo "Total batch size: $((BATCH_SIZE_PER_SESSION * NUM_SESSIONS))"
echo "Training data: All available (train_size=${TRAIN_SIZE})"
echo "GPUs: ${GPU_IDS[@]}"
echo "RR values: ${RR_VALUES[@]}"
echo "MLP hidden values: ${MLP_HIDDEN_VALUES[@]}"
echo "Stim embed dim values: ${EMBED_DIM_STIM_VALUES[@]}"
echo "Output directory: ${SWEEP_DIR}"
echo "=========================================="
echo ""

# Store model paths for TTA evaluation
MODEL_PATHS_FILE="${SWEEP_DIR}/model_paths.txt"
> "${MODEL_PATHS_FILE}"  # Clear file

# Track PIDs for each GPU
GPU_0_PID=""
GPU_1_PID=""

# Function to wait for any GPU to become available and return which one
wait_for_available_gpu() {
    while true; do
        # Check GPU 0
        if [ -z "$GPU_0_PID" ] || ! kill -0 "$GPU_0_PID" 2>/dev/null; then
            echo "0"
            return
        fi
        # Check GPU 1
        if [ -z "$GPU_1_PID" ] || ! kill -0 "$GPU_1_PID" 2>/dev/null; then
            echo "1"
            return
        fi
        # Both busy, wait and check again
        sleep 5
    done
}

# Phase 1: Train all models
echo "=========================================="
echo "PHASE 1: Training Models (Parallel on ${#GPU_IDS[@]} GPUs)"
echo "=========================================="
echo ""

job_num=0
total_jobs=$((${#RR_VALUES[@]} * ${#MLP_HIDDEN_VALUES[@]} * ${#EMBED_DIM_STIM_VALUES[@]}))

for rr in "${RR_VALUES[@]}"; do
    for mlp_hidden in "${MLP_HIDDEN_VALUES[@]}"; do
        for embed_dim_stim in "${EMBED_DIM_STIM_VALUES[@]}"; do
            job_num=$((job_num + 1))
            
            # Wait for an available GPU
            gpu=$(wait_for_available_gpu)
            
            echo "----------------------------------------"
            echo "Job ${job_num}/${total_jobs}: RR=${rr}, MLP_hidden=${mlp_hidden}, Embed_dim_stim=${embed_dim_stim} on GPU ${gpu}"
            echo "----------------------------------------"
            
            # Expected output directory from tma_standalone.py
            MODEL_NAME="${NUM_BASES}_${NUM_SESSIONS}_rr${rr}_inner_ts${TRAIN_SIZE}"
            if [ "${SHUFFLE}" = "true" ]; then
                MODEL_NAME="${MODEL_NAME}_shuffle"
            fi
            MODEL_NAME="${MODEL_NAME}_ld${LATENT_DIM}_bs$((BATCH_SIZE_PER_SESSION * NUM_SESSIONS))_mlp${mlp_hidden}_eds${embed_dim_stim}"
            
            SRC_DIR="test/${MODEL_NAME}"
            DEST_DIR="${MODEL_DIR}/${MODEL_NAME}"
            
            # Train model in background
            (
                python tma_standalone.py \
                    ${NUM_BASES} \
                    ${NUM_SESSIONS} \
                    ${gpu} \
                    ${COADAPT} \
                    ${rr} \
                    ${TRAIN_SIZE} \
                    ${SHUFFLE} \
                    --latent-dim ${LATENT_DIM} \
                    --batch-size-per-session ${BATCH_SIZE_PER_SESSION} \
                    --residual-mlp-hidden ${mlp_hidden} \
                    --embed-dim-stim ${embed_dim_stim}
                
                # Move model to sweep directory
                if [ -d "${SRC_DIR}" ]; then
                    mv "${SRC_DIR}" "${DEST_DIR}"
                    echo "[GPU ${gpu}] Moved model to: ${DEST_DIR}"
                    
                    # Record model path for TTA (with lock to prevent race conditions)
                    (
                        flock -x 200
                        echo "rr${rr}_mlp${mlp_hidden}_eds${embed_dim_stim}:${DEST_DIR}" >> "${MODEL_PATHS_FILE}"
                    ) 200>"${MODEL_PATHS_FILE}.lock"
                else
                    echo "[GPU ${gpu}] WARNING: Model directory not found: ${SRC_DIR}"
                fi
                
                echo "[GPU ${gpu}] Job complete: RR=${rr}, MLP_hidden=${mlp_hidden}, Embed_dim_stim=${embed_dim_stim}"
            ) &
            
            # Store PID for this GPU
            if [ "$gpu" == "0" ]; then
                GPU_0_PID=$!
                echo "Started training on GPU 0 (PID: ${GPU_0_PID})"
            else
                GPU_1_PID=$!
                echo "Started training on GPU 1 (PID: ${GPU_1_PID})"
            fi
            echo ""
        done
    done
done

# Wait for all remaining jobs to complete
echo "----------------------------------------"
echo "Waiting for all training jobs to complete..."
echo "----------------------------------------"
if [ -n "$GPU_0_PID" ]; then
    wait $GPU_0_PID 2>/dev/null || true
fi
if [ -n "$GPU_1_PID" ]; then
    wait $GPU_1_PID 2>/dev/null || true
fi

# Clean up lock file
rm -f "${MODEL_PATHS_FILE}.lock"

echo "=========================================="
echo "PHASE 1 COMPLETE: All models trained"
echo "=========================================="
echo ""

# Phase 2: Run TTA evaluation on all models
echo "=========================================="
echo "PHASE 2: TTA Evaluation (Multi-GPU)"
echo "=========================================="
echo ""

# Prepare model paths argument for tta_testing.py
MODEL_PATHS_ARGS=()
while IFS= read -r line; do
    MODEL_PATHS_ARGS+=("--model-paths" "${line}")
done < "${MODEL_PATHS_FILE}"

echo "Running TTA with the following models:"
cat "${MODEL_PATHS_FILE}"
echo ""

# Run TTA evaluation with multi-GPU support
python tta_testing.py \
    --use-multi-gpu \
    --gpu-ids "${GPU_IDS[@]}" \
    --support-sizes 500 \
    --max-adapt-sessions 5 \
    --tta-epochs 7001 \
    --tta-inner-steps 20 \
    --batch-size-per-session 500 \
    --output-dir "${TTA_RESULTS_DIR}" \
    --no-plot-display \
    "${MODEL_PATHS_ARGS[@]}"

echo ""
echo "=========================================="
echo "PHASE 2 COMPLETE: TTA evaluation finished"
echo "=========================================="
echo ""

# Summary
echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
echo "Models trained: ${#RR_VALUES[@]} × ${#MLP_HIDDEN_VALUES[@]} × ${#EMBED_DIM_STIM_VALUES[@]} = $((${#RR_VALUES[@]} * ${#MLP_HIDDEN_VALUES[@]} * ${#EMBED_DIM_STIM_VALUES[@]}))"
echo "GPUs used: ${GPU_IDS[@]}"
echo "Model directory: ${MODEL_DIR}"
echo "TTA results directory: ${TTA_RESULTS_DIR}"
echo "Model paths file: ${MODEL_PATHS_FILE}"
echo ""
echo "To analyze results:"
echo "  python plot_tta_results.py ${TTA_RESULTS_DIR}"
echo "=========================================="
