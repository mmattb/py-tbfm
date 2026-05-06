#!/bin/bash
# Train models with random folds of 20 sessions for cross-validation with TTA

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="random_folds_${TIMESTAMP}"

echo "Starting random folds training at ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_BASE}"

# Create output directory and timing log
mkdir -p ${OUTPUT_BASE}
TIMING_LOG="${OUTPUT_BASE}/timing_log.txt"
SESSION_LOG="${OUTPUT_BASE}/session_selections.txt"
METRICS_LOG="${OUTPUT_BASE}/fold_metrics.csv"

# Initialize logs
echo "Random Folds Training Log" > ${TIMING_LOG}
echo "Started at: ${TIMESTAMP}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}

echo "Session Selections for Each Fold" > ${SESSION_LOG}
echo "========================================" >> ${SESSION_LOG}
echo "" >> ${SESSION_LOG}

echo "fold,seed,start_time,end_time,duration_sec,final_train_r2,final_test_r2,status" > ${METRICS_LOG}

# Training parameters
NUM_FOLDS=20  # Number of random folds to create
NUM_SESSIONS=20  # Number of sessions per fold
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

# Train models for each fold using both GPUs (2 at a time max)
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    GPU_ID=$((i % 2))

    # Wait for the specific GPU to be free if it's occupied
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        PREV_FOLD=${FOLD_GPUS[$GPU_ID]}
        echo "GPU ${GPU_ID} is occupied, waiting for fold ${PREV_FOLD} (PID ${GPU_PIDS[$GPU_ID]}) to finish..."
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

        # Extract metrics from output directory
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

        # Force GPU memory cleanup
        python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
        sleep 2
    fi

    echo "========================================="
    echo "Starting training for fold ${i} (${NUM_SESSIONS} random sessions) on GPU ${GPU_ID}"
    echo "========================================="

    OUT_DIR="${OUTPUT_BASE}/fold${i}"
    START_TIME=$(date +%s)
    START_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Record start time and GPU assignment
    FOLD_START_TIMES[$i]=$START_TIME
    FOLD_GPUS[$GPU_ID]=$i

    # Log which sessions will be selected (preview using Python)
    SESSIONS=$(python -c "
import random
random.seed($i)
all_sessions = [
    'MonkeyG_20150914_Session1_S1', 'MonkeyG_20150914_Session3_S1',
    'MonkeyG_20150915_Session2_S1', 'MonkeyG_20150915_Session3_S1',
    'MonkeyG_20150915_Session4_S1', 'MonkeyG_20150915_Session5_S1',
    'MonkeyG_20150916_Session4_S1', 'MonkeyG_20150917_Session1_M1',
    'MonkeyG_20150917_Session1_S1', 'MonkeyG_20150917_Session2_M1',
    'MonkeyG_20150917_Session2_S1', 'MonkeyG_20150917_Session3_M1',
    'MonkeyG_20150917_Session3_S1', 'MonkeyG_20150918_Session1_M1',
    'MonkeyG_20150918_Session1_S1', 'MonkeyG_20150921_Session3_S1',
    'MonkeyG_20150921_Session5_S1', 'MonkeyG_20150922_Session1_S1',
    'MonkeyG_20150922_Session2_S1', 'MonkeyG_20150922_Session3_S1',
    'MonkeyG_20150925_Session1_S1', 'MonkeyG_20150925_Session2_S1',
    'MonkeyJ_20160426_Session1_S1', 'MonkeyJ_20160426_Session2_S1',
    'MonkeyJ_20160426_Session3_S1', 'MonkeyJ_20160428_Session2_S1',
    'MonkeyJ_20160428_Session3_S1', 'MonkeyJ_20160429_Session1_S1',
    'MonkeyJ_20160429_Session3_S1', 'MonkeyJ_20160502_Session1_S1',
    'MonkeyJ_20160624_Session3_S1', 'MonkeyJ_20160624_Session4_S1',
    'MonkeyJ_20160625_Session4_S1', 'MonkeyJ_20160625_Session5_S1',
    'MonkeyJ_20160627_Session1_S1', 'MonkeyJ_20160627_Session2_S1',
    'MonkeyJ_20160630_Session1_S1', 'MonkeyJ_20160630_Session3_S1',
    'MonkeyJ_20160702_Session2_S1', 'MonkeyJ_20160702_Session4_S1',
]
selected = random.sample(all_sessions, ${NUM_SESSIONS})
print(','.join(selected))
")

    echo "Fold ${i} (seed: ${i}):" >> ${SESSION_LOG}
    echo "$SESSIONS" | tr ',' '\n' | sed 's/^/  - /' >> ${SESSION_LOG}
    echo "" >> ${SESSION_LOG}

    # Use different random seeds for each fold to ensure different session selections
    python tma_standalone.py \
        ${NUM_BASES} \
        ${NUM_SESSIONS} \
        ${GPU_ID} \
        false \
        16 \
        ${TRAIN_SIZE} \
        true \
        --latent-dim ${LATENT_DIM} \
        --batch-size-per-session ${BATCH_SIZE} \
        --out-dir ${OUT_DIR} \
        --random-seed $i &

    PID=$!
    GPU_PIDS[$GPU_ID]=$PID
    echo "Started training for fold ${i} on GPU ${GPU_ID} (PID: ${PID}, seed: ${i}) at ${START_TIMESTAMP}"
    echo "Fold ${i}: STARTED at ${START_TIMESTAMP} on GPU ${GPU_ID}" >> ${TIMING_LOG}
    echo ""
done

# Wait for all remaining training jobs to complete
echo "Waiting for all remaining training jobs to complete..."
echo "" >> ${TIMING_LOG}

# Process completion of final GPU 0 job if it exists
if [ -n "${GPU_PIDS[0]}" ]; then
    FOLD=${FOLD_GPUS[0]}
    wait ${GPU_PIDS[0]}
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

# Process completion of final GPU 1 job if it exists
if [ -n "${GPU_PIDS[1]}" ]; then
    FOLD=${FOLD_GPUS[1]}
    wait ${GPU_PIDS[1]}
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

echo "All training jobs completed successfully"
COMPLETION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "" >> ${TIMING_LOG}
echo "All folds completed: ${COMPLETION_TIMESTAMP}" >> ${TIMING_LOG}
echo ""

# Force GPU memory cleanup after all training
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Generate summary statistics
SUMMARY_STATS=$(python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${METRICS_LOG}')

    # Filter successful runs
    success_df = df[df['status'] == 'SUCCESS']

    if len(success_df) > 0:
        # Compute statistics
        avg_duration = success_df['duration_sec'].mean()
        std_duration = success_df['duration_sec'].std()
        min_duration = success_df['duration_sec'].min()
        max_duration = success_df['duration_sec'].max()

        # Handle R² metrics
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
echo "Random folds training complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Number of folds: ${NUM_FOLDS}"
echo "Sessions per fold: ${NUM_SESSIONS}"
echo "========================================="
echo ""

# Display summary statistics
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

# Write final summary to timing log
echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Training Configuration:" >> ${TIMING_LOG}
echo "  Number of folds: ${NUM_FOLDS}" >> ${TIMING_LOG}
echo "  Sessions per fold: ${NUM_SESSIONS}" >> ${TIMING_LOG}
echo "  Latent dim: ${LATENT_DIM}" >> ${TIMING_LOG}
echo "  Num bases: ${NUM_BASES}" >> ${TIMING_LOG}
echo "  Train size: ${TRAIN_SIZE}" >> ${TIMING_LOG}
echo "  Batch size per session: ${BATCH_SIZE}" >> ${TIMING_LOG}
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

# Display timing summary
echo "Full timing log:"
cat ${TIMING_LOG}
