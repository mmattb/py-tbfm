#!/bin/bash
# Run TTA evaluation (co-adaptation strategy) on all held-out sessions for each fold
# from a coadapt random-folds training run.
#
# Usage: $0 <path_to_coadapt_random_folds_results> [existing_tta_output_dir]
# Example: $0 random_folds_coadapt_20260201_120000
# Example (resume): $0 random_folds_coadapt_20260201_120000 random_folds_coadapt_20260201_120000/tta_results_500_20260201_130000

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_coadapt_random_folds_results> [existing_tta_output_dir]"
    echo "Example: $0 random_folds_coadapt_20260201_120000"
    echo "Example (resume): $0 random_folds_coadapt_20260201_120000 random_folds_coadapt_20260201_120000/tta_results_500_20260201_130000"
    exit 1
fi

FOLDS_DIR="$1"
RESUME_DIR="$2"

if [ ! -d "$FOLDS_DIR" ]; then
    echo "ERROR: Directory not found: $FOLDS_DIR"
    exit 1
fi

echo "========================================="
echo "TTA Cross-Validation on Coadapt Random Folds"
echo "========================================="
echo "Folds directory: $FOLDS_DIR"
echo ""

# TTA parameters
SUPPORT_SIZE=${SUPPORT_SIZE:-1000}
TTA_EPOCHS=7001
NUM_FOLDS=20

# Create or reuse TTA output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo "ERROR: Resume directory not found: $RESUME_DIR"
        exit 1
    fi
    TTA_OUTPUT_DIR="$RESUME_DIR"
    echo "Resuming previous run from: ${TTA_OUTPUT_DIR}"
else
    TTA_OUTPUT_DIR="${FOLDS_DIR}/tta_results_${SUPPORT_SIZE}_${TIMESTAMP}"
    mkdir -p ${TTA_OUTPUT_DIR}
fi

# Create logs
TIMING_LOG="${TTA_OUTPUT_DIR}/tta_timing_log.txt"
RESULTS_SUMMARY="${TTA_OUTPUT_DIR}/tta_summary.csv"

echo "TTA Coadapt Cross-Validation Log" > ${TIMING_LOG}
echo "Started at: ${TIMESTAMP}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}

echo "fold,session_id,strategy,support_size,r2,start_time,end_time,duration_sec,status" > ${RESULTS_SUMMARY}


# Count available folds
AVAILABLE_FOLDS=0
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${FOLDS_DIR}/fold${i}"
    if [ -d "$FOLD_DIR" ]; then
        AVAILABLE_FOLDS=$((AVAILABLE_FOLDS + 1))
    fi
done

echo "Found ${AVAILABLE_FOLDS} folds to process"
echo "Support size: ${SUPPORT_SIZE}"
echo "TTA epochs: ${TTA_EPOCHS}"
echo "Output directory: ${TTA_OUTPUT_DIR}"
echo ""

# Process each fold
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${FOLDS_DIR}/fold${i}"

    if [ ! -d "$FOLD_DIR" ]; then
        echo "WARNING: Fold ${i} directory not found, skipping..."
        continue
    fi

    # Check if model files exist (new save format: tbfm.torch in fold dir)
    if [ ! -f "${FOLD_DIR}/tbfm.torch" ] && [ ! -f "${FOLD_DIR}/model.torch" ] && [ ! -f "${FOLD_DIR}/model_nf_1.torch" ]; then
        echo "WARNING: No model file found in fold ${i}, skipping..."
        continue
    fi

    # Check if hisi file exists
    if [ ! -f "${FOLD_DIR}/hisi_nf_1.torch" ] && [ ! -f "${FOLD_DIR}/hisi.torch" ]; then
        echo "WARNING: No hisi file found in fold ${i}, skipping..."
        continue
    fi

    # Check if this fold was already completed
    FOLD_TTA_DIR="${TTA_OUTPUT_DIR}/fold${i}"
    if [ -d "$FOLD_TTA_DIR" ] && ls ${FOLD_TTA_DIR}/tta_support_*.json 1>/dev/null 2>&1; then
        echo "Fold ${i} already completed (results found in ${FOLD_TTA_DIR}), skipping..."
        continue
    fi

    echo "========================================="
    echo "Processing Fold ${i}"
    echo "========================================="

    FOLD_START=$(date +%s)
    FOLD_START_TS=$(date +%Y%m%d_%H%M%S)

    echo "Fold ${i}: STARTED at ${FOLD_START_TS}" >> ${TIMING_LOG}

    mkdir -p ${FOLD_TTA_DIR}

    # Run TTA for this fold using co-adaptation strategy
    python tta_testing.py \
        --model-paths fold${i}:${FOLD_DIR} \
        --use-multi-gpu \
        --gpu-ids 0 1 \
        --support-sizes ${SUPPORT_SIZE} \
        --max-adapt-sessions 20 \
        --tta-epochs ${TTA_EPOCHS} \
        --output-dir ${FOLD_TTA_DIR} \
        --unfreeze-bases \
        --progressive-unfreezing-threshold 0 \
        --coadapt-tta

    EXIT_CODE=$?
    FOLD_END=$(date +%s)
    FOLD_DURATION=$((FOLD_END - FOLD_START))
    FOLD_END_TS=$(date +%Y%m%d_%H%M%S)

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: TTA failed for fold ${i}"
        echo "Fold ${i}: FAILED at ${FOLD_END_TS} (duration: ${FOLD_DURATION}s)" >> ${TIMING_LOG}
        echo "${i},ALL,ALL,${SUPPORT_SIZE},NA,${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},FAILED" >> ${RESULTS_SUMMARY}
        echo "Continuing with remaining folds..."
        echo ""
        continue
    fi

    echo "Fold ${i}: COMPLETED at ${FOLD_END_TS} (duration: ${FOLD_DURATION}s)" >> ${TIMING_LOG}

    # Extract per-session results
    RESULTS_JSON=$(ls -t ${FOLD_TTA_DIR}/tta_support_*.json 2>/dev/null | head -n 1)

    if [ -f "$RESULTS_JSON" ]; then
        echo "  Extracting results from: $(basename $RESULTS_JSON)"

        python -c "
import json
import sys

try:
    with open('${RESULTS_JSON}', 'r') as f:
        results = json.load(f)

    for run in results.get('runs', []):
        strategy = run.get('strategy', 'unknown')
        support_size = run.get('support_size', 0)
        per_session_r2s = run.get('per_session_r2s', {})

        if per_session_r2s:
            for session_id, r2 in per_session_r2s.items():
                print(f'${i},{session_id},{strategy},{support_size},{r2},${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},SUCCESS')
        else:
            overall_r2 = run.get('r2', 'NA')
            print(f'${i},OVERALL,{strategy},{support_size},{overall_r2},${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},SUCCESS')
except Exception as e:
    print(f'ERROR: Failed to parse results: {e}', file=sys.stderr)
    sys.exit(1)
" >> ${RESULTS_SUMMARY}

        if [ $? -ne 0 ]; then
            echo "  WARNING: Failed to extract per-session results"
        fi
    else
        echo "  WARNING: No results JSON file found"
        echo "${i},ALL,ALL,${SUPPORT_SIZE},NA,${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},NO_RESULTS" >> ${RESULTS_SUMMARY}
    fi

    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    sleep 2

    echo "Fold ${i} complete (${FOLD_DURATION}s)"
    echo ""
done

# Generate summary statistics
COMPLETION_TS=$(date +%Y%m%d_%H%M%S)
echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "All folds completed: ${COMPLETION_TS}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

SUMMARY_STATS=$(python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${RESULTS_SUMMARY}')
    success_df = df[df['status'] == 'SUCCESS']

    if len(success_df) > 0:
        unique_folds = success_df['fold'].nunique()
        total_sessions = len(success_df)

        strategies = success_df['strategy'].unique()

        print(f'Processed {unique_folds} folds with {total_sessions} total session results')
        print('')
        print('Per-strategy statistics:')

        for strategy in strategies:
            strategy_df = success_df[success_df['strategy'] == strategy]
            r2_vals = pd.to_numeric(strategy_df['r2'], errors='coerce')

            if not r2_vals.isna().all():
                mean_r2 = r2_vals.mean()
                std_r2 = r2_vals.std()
                min_r2 = r2_vals.min()
                max_r2 = r2_vals.max()
                count = len(r2_vals.dropna())

                print(f'  {strategy}:')
                print(f'    Sessions: {count}')
                print(f'    Mean R²: {mean_r2:.4f} ± {std_r2:.4f}')
                print(f'    Range: [{min_r2:.4f}, {max_r2:.4f}]')
                print('')

        print('Per-fold average R²:')
        fold_avg = success_df.groupby('fold')['r2'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
        for fold, avg_r2 in fold_avg.items():
            print(f'  Fold {fold}: {avg_r2:.4f}')

        print('')
        print(f'Overall mean R²: {pd.to_numeric(success_df[\"r2\"], errors=\"coerce\").mean():.4f}')
    else:
        print('No successful results found')
except Exception as e:
    print(f'Error computing summary: {e}')
" 2>/dev/null)

echo "========================================="
echo "TTA Coadapt Cross-Validation Complete!"
echo "========================================="
echo "Results directory: ${TTA_OUTPUT_DIR}"
echo ""
echo "$SUMMARY_STATS"
echo ""
echo "Output files:"
echo "  - ${TIMING_LOG}: Detailed timing log"
echo "  - ${RESULTS_SUMMARY}: Per-session results CSV"
echo ""
echo "Per-fold results saved to:"
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_TTA_DIR="${TTA_OUTPUT_DIR}/fold${i}"
    if [ -d "$FOLD_TTA_DIR" ]; then
        echo "  - ${FOLD_TTA_DIR}/"
    fi
done
echo ""

echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Summary Statistics" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "$SUMMARY_STATS" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Completed at: ${COMPLETION_TS}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

echo "Full timing log:"
cat ${TIMING_LOG}
