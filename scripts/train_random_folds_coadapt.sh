#!/bin/bash
# Train coadapt models using the exact same folds as an existing MAML random-folds run.
# Reads hisi.torch from each MAML fold to guarantee identical session selections.
#
# Usage: ./train_random_folds_coadapt.sh <maml_folds_dir>
# Example: ./train_random_folds_coadapt.sh random_folds_20260101_211200

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_maml_random_folds_dir>"
    echo "Example: $0 random_folds_20260101_211200"
    exit 1
fi

MAML_FOLDS_DIR="$1"

if [ ! -d "$MAML_FOLDS_DIR" ]; then
    echo "ERROR: Directory not found: $MAML_FOLDS_DIR"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="random_folds_coadapt_${TIMESTAMP}"

echo "Starting random folds coadapt training at ${TIMESTAMP}"
echo "MAML reference dir: ${MAML_FOLDS_DIR}"
echo "Output directory: ${OUTPUT_BASE}"

mkdir -p ${OUTPUT_BASE}
TIMING_LOG="${OUTPUT_BASE}/timing_log.txt"
SESSION_LOG="${OUTPUT_BASE}/session_selections.txt"
METRICS_LOG="${OUTPUT_BASE}/fold_metrics.csv"

echo "Random Folds Coadapt Training Log" > ${TIMING_LOG}
echo "Started at: ${TIMESTAMP}" >> ${TIMING_LOG}
echo "MAML reference dir: ${MAML_FOLDS_DIR}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}

echo "Session Selections for Each Fold" > ${SESSION_LOG}
echo "(Sourced from ${MAML_FOLDS_DIR})" >> ${SESSION_LOG}
echo "========================================" >> ${SESSION_LOG}
echo "" >> ${SESSION_LOG}

echo "fold,seed,start_time,end_time,duration_sec,final_train_r2,final_test_r2,status" > ${METRICS_LOG}

# Training parameters (must match original MAML run)
NUM_FOLDS=20
NUM_SESSIONS=20
LATENT_DIM=96
NUM_BASES=100
TRAIN_SIZE=5000
BATCH_SIZE=500

# Track PIDs and start times for each GPU
declare -A GPU_PIDS
declare -A FOLD_START_TIMES
declare -A FOLD_GPUS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

for i in $(seq 0 $((NUM_FOLDS - 1))); do
    MAML_FOLD_DIR="${MAML_FOLDS_DIR}/fold${i}"

    # Verify reference fold exists
    if [ ! -d "$MAML_FOLD_DIR" ]; then
        echo "ERROR: MAML fold ${i} not found at ${MAML_FOLD_DIR}"
        exit 1
    fi

    # Determine which hisi file to read (nf_1 variant takes priority if present)
    if [ -f "${MAML_FOLD_DIR}/hisi_nf_1.torch" ]; then
        HISI_FILE="${MAML_FOLD_DIR}/hisi_nf_1.torch"
    elif [ -f "${MAML_FOLD_DIR}/hisi.torch" ]; then
        HISI_FILE="${MAML_FOLD_DIR}/hisi.torch"
    else
        echo "ERROR: No hisi file found in ${MAML_FOLD_DIR}"
        exit 1
    fi

    # Read session IDs from the hisi file
    SESSIONS=$(python -c "
import torch, sys
try:
    ids = torch.load('${HISI_FILE}', map_location='cpu', weights_only=False)
    print(','.join(ids))
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read ${HISI_FILE}"
        exit 1
    fi

    GPU_ID=$((i % 2))

    # Wait for the specific GPU to be free if occupied
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        PREV_FOLD=${FOLD_GPUS[$GPU_ID]}
        echo "GPU ${GPU_ID} occupied, waiting for fold ${PREV_FOLD} (PID ${GPU_PIDS[$GPU_ID]}) to finish..."
        wait ${GPU_PIDS[$GPU_ID]}
        EXIT_CODE=$?

        END_TIME=$(date +%s)
        START_TIME=${FOLD_START_TIMES[$PREV_FOLD]}
        DURATION=$((END_TIME - START_TIME))
        END_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "ERROR: Training failed for fold ${PREV_FOLD} on GPU ${GPU_ID}"
            echo "Fold ${PREV_FOLD}: FAILED at ${END_TIMESTAMP} (duration: ${DURATION}s)" >> ${TIMING_LOG}
            echo "${PREV_FOLD},${PREV_FOLD},$(date -d @${START_TIME} +%Y%m%d_%H%M%S),${END_TIMESTAMP},${DURATION},NA,NA,FAILED" >> ${METRICS_LOG}
            exit 1
        fi

        OUT_DIR="${OUTPUT_BASE}/fold${PREV_FOLD}"
        TRAIN_R2="NA"
        TEST_R2="NA"
        if [ -f "${OUT_DIR}/results.pkl" ]; then
            METRICS=$(python -c "
import pickle
try:
    with open('${OUT_DIR}/results.pkl', 'rb') as f:
        results = pickle.load(f)
    train_r2 = results.get('train_r2s', [[None, 'NA']])[-1][1]
    test_r2 = results.get('test_r2s', [[None, 'NA']])[-1][1]
    print(f'{train_r2},{test_r2}')
except:
    print('NA,NA')
" 2>/dev/null)
            TRAIN_R2=$(echo $METRICS | cut -d',' -f1)
            TEST_R2=$(echo $METRICS | cut -d',' -f2)
        fi

        echo "Fold ${PREV_FOLD}: COMPLETED at ${END_TIMESTAMP} (duration: ${DURATION}s, train_r2: ${TRAIN_R2}, test_r2: ${TEST_R2})" >> ${TIMING_LOG}
        echo "${PREV_FOLD},${PREV_FOLD},$(date -d @${START_TIME} +%Y%m%d_%H%M%S),${END_TIMESTAMP},${DURATION},${TRAIN_R2},${TEST_R2},SUCCESS" >> ${METRICS_LOG}
        echo "GPU ${GPU_ID} freed, fold ${PREV_FOLD} completed in ${DURATION}s"

        python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
        sleep 2
    fi

    echo "========================================="
    echo "Starting coadapt training for fold ${i} on GPU ${GPU_ID}"
    echo "Sessions: ${SESSIONS}"
    echo "========================================="

    OUT_DIR="${OUTPUT_BASE}/fold${i}"
    START_TIME=$(date +%s)
    START_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    FOLD_START_TIMES[$i]=$START_TIME
    FOLD_GPUS[$GPU_ID]=$i

    echo "Fold ${i}:" >> ${SESSION_LOG}
    echo "$SESSIONS" | tr ',' '\n' | sed 's/^/  - /' >> ${SESSION_LOG}
    echo "" >> ${SESSION_LOG}

    python tma_standalone.py \
        ${NUM_BASES} \
        ${NUM_SESSIONS} \
        ${GPU_ID} \
        true \
        16 \
        ${TRAIN_SIZE} \
        true \
        --latent-dim ${LATENT_DIM} \
        --batch-size-per-session ${BATCH_SIZE} \
        --out-dir ${OUT_DIR} \
        --held-in-sessions ${SESSIONS} &

    PID=$!
    GPU_PIDS[$GPU_ID]=$PID
    echo "Started coadapt training for fold ${i} on GPU ${GPU_ID} (PID: ${PID}) at ${START_TIMESTAMP}"
    echo "Fold ${i}: STARTED at ${START_TIMESTAMP} on GPU ${GPU_ID}" >> ${TIMING_LOG}
    echo ""
done

# Wait for all remaining jobs
echo "Waiting for all remaining training jobs to complete..."
echo "" >> ${TIMING_LOG}

for GPU_ID in 0 1; do
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        FOLD=${FOLD_GPUS[$GPU_ID]}
        wait ${GPU_PIDS[$GPU_ID]}
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        START_TIME=${FOLD_START_TIMES[$FOLD]}
        DURATION=$((END_TIME - START_TIME))
        END_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "ERROR: Training failed for fold ${FOLD}"
            echo "Fold ${FOLD}: FAILED at ${END_TIMESTAMP}" >> ${TIMING_LOG}
            echo "${FOLD},${FOLD},$(date -d @${START_TIME} +%Y%m%d_%H%M%S),${END_TIMESTAMP},${DURATION},NA,NA,FAILED" >> ${METRICS_LOG}
            exit 1
        fi

        OUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
        TRAIN_R2="NA"
        TEST_R2="NA"
        if [ -f "${OUT_DIR}/results.pkl" ]; then
            METRICS=$(python -c "
import pickle
try:
    with open('${OUT_DIR}/results.pkl', 'rb') as f:
        results = pickle.load(f)
    train_r2 = results.get('train_r2s', [[None, 'NA']])[-1][1]
    test_r2 = results.get('test_r2s', [[None, 'NA']])[-1][1]
    print(f'{train_r2},{test_r2}')
except:
    print('NA,NA')
" 2>/dev/null)
            TRAIN_R2=$(echo $METRICS | cut -d',' -f1)
            TEST_R2=$(echo $METRICS | cut -d',' -f2)
        fi

        echo "Fold ${FOLD}: COMPLETED at ${END_TIMESTAMP} (duration: ${DURATION}s, train_r2: ${TRAIN_R2}, test_r2: ${TEST_R2})" >> ${TIMING_LOG}
        echo "${FOLD},${FOLD},$(date -d @${START_TIME} +%Y%m%d_%H%M%S),${END_TIMESTAMP},${DURATION},${TRAIN_R2},${TEST_R2},SUCCESS" >> ${METRICS_LOG}
    fi
done

echo "All training jobs completed successfully"
COMPLETION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "" >> ${TIMING_LOG}
echo "All folds completed: ${COMPLETION_TIMESTAMP}" >> ${TIMING_LOG}
echo ""

python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Generate summary statistics
SUMMARY_STATS=$(python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${METRICS_LOG}')
    success_df = df[df['status'] == 'SUCCESS']

    if len(success_df) > 0:
        avg_duration = success_df['duration_sec'].mean()
        std_duration = success_df['duration_sec'].std()
        min_duration = success_df['duration_sec'].min()
        max_duration = success_df['duration_sec'].max()

        train_r2_vals = pd.to_numeric(success_df['final_train_r2'], errors='coerce')
        test_r2_vals = pd.to_numeric(success_df['final_test_r2'], errors='coerce')

        train_r2_mean = train_r2_vals.mean() if not train_r2_vals.isna().all() else 'NA'
        train_r2_std = train_r2_vals.std() if not train_r2_vals.isna().all() else 'NA'
        test_r2_mean = test_r2_vals.mean() if not test_r2_vals.isna().all() else 'NA'
        test_r2_std = test_r2_vals.std() if not test_r2_vals.isna().all() else 'NA'

        print(f'SUCCESS:{len(success_df)}:{avg_duration:.1f}:{std_duration:.1f}:{min_duration:.1f}:{max_duration:.1f}:{train_r2_mean}:{train_r2_std}:{test_r2_mean}:{test_r2_std}')
    else:
        print('NOSUCCESS')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null)

echo "========================================="
echo "Random folds coadapt training complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Number of folds: ${NUM_FOLDS}"
echo "Sessions per fold: ${NUM_SESSIONS}"
echo "========================================="
echo ""

if [[ $SUMMARY_STATS == SUCCESS:* ]]; then
    IFS=':' read -ra STATS <<< "$SUMMARY_STATS"
    echo "Summary Statistics:"
    echo "  Successful folds: ${STATS[1]}/${NUM_FOLDS}"
    echo "  Training duration: ${STATS[2]}s ± ${STATS[3]}s (min: ${STATS[4]}s, max: ${STATS[5]}s)"
    if [[ ${STATS[6]} != "NA" ]]; then
        echo "  Final train R²: ${STATS[6]} ± ${STATS[7]}"
    fi
    if [[ ${STATS[8]} != "NA" ]]; then
        echo "  Final test R²: ${STATS[8]} ± ${STATS[9]}"
    fi
    echo ""
fi

echo "Output files:"
echo "  - ${TIMING_LOG}: Detailed timing log"
echo "  - ${SESSION_LOG}: Session selections for each fold"
echo "  - ${METRICS_LOG}: Metrics CSV for analysis"
echo ""

echo "Model paths for TTA testing:"
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    echo "  fold${i}:${OUTPUT_BASE}/fold${i}"
done
echo ""

echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Training Configuration:" >> ${TIMING_LOG}
echo "  Number of folds: ${NUM_FOLDS}" >> ${TIMING_LOG}
echo "  Sessions per fold: ${NUM_SESSIONS}" >> ${TIMING_LOG}
echo "  Latent dim: ${LATENT_DIM}" >> ${TIMING_LOG}
echo "  Num bases: ${NUM_BASES}" >> ${TIMING_LOG}
echo "  Train size: ${TRAIN_SIZE}" >> ${TIMING_LOG}
echo "  Batch size per session: ${BATCH_SIZE}" >> ${TIMING_LOG}
echo "  Coadapt: true" >> ${TIMING_LOG}
echo "  MAML reference: ${MAML_FOLDS_DIR}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

if [[ $SUMMARY_STATS == SUCCESS:* ]]; then
    echo "" >> ${TIMING_LOG}
    echo "Summary Statistics:" >> ${TIMING_LOG}
    echo "  Successful folds: ${STATS[1]}/${NUM_FOLDS}" >> ${TIMING_LOG}
    echo "  Training duration: ${STATS[2]}s ± ${STATS[3]}s (min: ${STATS[4]}s, max: ${STATS[5]}s)" >> ${TIMING_LOG}
    if [[ ${STATS[6]} != "NA" ]]; then
        echo "  Final train R²: ${STATS[6]} ± ${STATS[7]}" >> ${TIMING_LOG}
    fi
    if [[ ${STATS[8]} != "NA" ]]; then
        echo "  Final test R²: ${STATS[8]} ± ${STATS[9]}" >> ${TIMING_LOG}
    fi
fi

echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Completed at: ${COMPLETION_TIMESTAMP}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

echo "Full timing log:"
cat ${TIMING_LOG}
