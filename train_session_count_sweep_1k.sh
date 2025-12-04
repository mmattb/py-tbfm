#!/bin/bash
# Train models with different session counts and run TTA evaluation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="session_count_sweep_${TIMESTAMP}"

echo "Starting session count sweep at ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_BASE}"

# Create output directory and timing log
mkdir -p ${OUTPUT_BASE}
TIMING_LOG="${OUTPUT_BASE}/timing_log.txt"
echo "Session Count Sweep Timing Log" > ${TIMING_LOG}
echo "Started at: ${TIMESTAMP}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}

# Session counts to test
SESSION_COUNTS=(1 5 10 15 20 25)
LATENT_DIM=96
NUM_BASES=100
TRAIN_SIZE=1000
BATCH_SIZE=500

# Track PIDs for each GPU
declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

# Train models for each session count using both GPUs (2 at a time max)
for i in "${!SESSION_COUNTS[@]}"; do
    NS=${SESSION_COUNTS[$i]}
    GPU_ID=$((i % 2))
    
    # Wait for the specific GPU to be free if it's occupied
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        echo "GPU ${GPU_ID} is occupied, waiting for PID ${GPU_PIDS[$GPU_ID]} to finish..."
        wait ${GPU_PIDS[$GPU_ID]}
        EXIT_CODE=$?
        
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "ERROR: Training failed for model on GPU ${GPU_ID}"
            echo "Train failed: $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
            exit 1
        fi
        
        echo "Train complete: $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
        echo "GPU ${GPU_ID} freed, continuing..."
        
        # Force GPU memory cleanup
        python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
        sleep 2
    fi
    
    echo "========================================="
    echo "Starting training for num_sessions=${NS} on GPU ${GPU_ID}"
    echo "========================================="
    
    OUT_DIR="${OUTPUT_BASE}/1kx${NS}_maml"
    
    python tma_standalone.py \
        ${NUM_BASES} \
        ${NS} \
        ${GPU_ID} \
        false \
        16 \
        ${TRAIN_SIZE} \
        true \
        --latent-dim ${LATENT_DIM} \
        --batch-size-per-session ${BATCH_SIZE} \
        --out-dir ${OUT_DIR} &
    
    PID=$!
    GPU_PIDS[$GPU_ID]=$PID
    echo "Started training for num_sessions=${NS} on GPU ${GPU_ID} (PID: ${PID})"
    echo "Train start (1kx${NS}_maml): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
    echo ""
done

# Wait for all remaining training jobs to complete
echo "Waiting for all remaining training jobs to complete..."
wait
EXIT_CODE=$?
TRAIN_END=$(date +%s)

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Training failed for one of the models"
    echo "Train failed: $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
    exit 1
fi

echo "All MAML training jobs completed successfully"
echo "All MAML models trained: $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
echo ""

# Force GPU memory cleanup after all training
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
sleep 5

# Verify GPUs are clear before proceeding
echo "Verifying GPU memory is cleared..."
for i in {1..30}; do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print $1}')
    if [ "$GPU_MEM" -lt 1000 ]; then
        echo "GPU 0 cleared (${GPU_MEM}MB used)"
        break
    fi
    echo "Waiting for GPU 0 to clear... (${GPU_MEM}MB used, attempt $i/30)"
    sleep 2
done

# Train coadapt model with 25 sessions on GPU 0
echo "========================================="
echo "Starting training for coadapt model (25 sessions) on GPU 0"
echo "========================================="

OUT_DIR_COADAPT="${OUTPUT_BASE}/1kx25_coadapt"
COADAPT_TRAIN_START=$(date +%s)
echo "Train start (1kx25_coadapt): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}

python tma_standalone.py \
    ${NUM_BASES} \
    25 \
    0 \
    true \
    16 \
    ${TRAIN_SIZE} \
    true \
    --latent-dim ${LATENT_DIM} \
    --batch-size-per-session ${BATCH_SIZE} \
    --out-dir ${OUT_DIR_COADAPT}

EXIT_CODE=$?
COADAPT_TRAIN_END=$(date +%s)
COADAPT_TRAIN_TIME=$((COADAPT_TRAIN_END - COADAPT_TRAIN_START))

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Coadapt training failed"
    echo "Train failed (1kx25_coadapt): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
    exit 1
fi

echo "Coadapt training completed successfully"
echo "Train complete (1kx25_coadapt): $(date +%Y%m%d_%H%M%S), Duration: ${COADAPT_TRAIN_TIME}s" >> ${TIMING_LOG}
echo ""

# Force GPU memory cleanup
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

echo "========================================="
echo "All models trained. Starting TTA evaluation..."
echo "========================================="

# Run TTA sweep on 25 session models only
MODEL_PATHS_ARG="1kx25_coadapt:${OUTPUT_BASE}/1kx25_coadapt"
MODEL_PATHS_ARG="${MODEL_PATHS_ARG} 1kx25_maml:${OUTPUT_BASE}/1kx25_maml"

TTA_25_START=$(date +%s)
echo "TTA start (25-session models sweep): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}

python tta_testing.py \
    --model-paths ${MODEL_PATHS_ARG} \
    --use-multi-gpu \
    --gpu-ids 0 1 \
    --support-sizes 50 100 250 500 1000 2500 5000 \
    --max-adapt-sessions 15 \
    --tta-epochs 7001 \
    --output-dir ${OUTPUT_BASE}/tta_results \
    --unfreeze-bases \
    --progressive-unfreezing-threshold 0 \
    --include-vanilla-tbfm


EXIT_CODE=$?
TTA_25_END=$(date +%s)
TTA_25_TIME=$((TTA_25_END - TTA_25_START))

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: TTA sweep failed"
    echo "TTA failed (25-session sweep): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
    exit 1
fi

echo "TTA complete (25-session sweep): $(date +%Y%m%d_%H%M%S), Duration: ${TTA_25_TIME}s" >> ${TIMING_LOG}

# Force GPU memory cleanup
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Run TTA evaluation on trained models up to 20 sessions
MODEL_PATHS_ARG=""
for NS in "${SESSION_COUNTS[@]}"; do
    if [ ${NS} -ne 25 ]; then
        MODEL_PATHS_ARG="${MODEL_PATHS_ARG} 1kx${NS}_maml:${OUTPUT_BASE}/1kx${NS}_maml"
    fi
done

TTA_MAIN_START=$(date +%s)
echo "TTA start (main evaluation 1-20 sessions): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}

python tta_testing.py \
    --model-paths ${MODEL_PATHS_ARG} \
    --use-multi-gpu \
    --gpu-ids 0 1 \
    --support-sizes 2500 \
    --max-adapt-sessions 15 \
    --tta-epochs 7001 \
    --output-dir ${OUTPUT_BASE}/tta_results \
    --unfreeze-bases \
    --progressive-unfreezing-threshold 0


EXIT_CODE=$?
TTA_MAIN_END=$(date +%s)
TTA_MAIN_TIME=$((TTA_MAIN_END - TTA_MAIN_START))

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: TTA evaluation failed"
    echo "TTA failed (main evaluation): $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
    exit 1
fi

echo "TTA complete (main evaluation): $(date +%Y%m%d_%H%M%S), Duration: ${TTA_MAIN_TIME}s" >> ${TIMING_LOG}

# Force GPU memory cleanup
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"


echo "========================================="
echo "Session count sweep complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "========================================="

# Write final summary to timing log
TOTAL_END=$(date +%s)
echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Completed at: $(date +%Y%m%d_%H%M%S)" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

# Display timing summary
echo ""
echo "Timing Summary:"
cat ${TIMING_LOG}
