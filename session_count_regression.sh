#!/bin/bash
# Session Count Regression Analysis
# Trains models on different session counts and evaluates TTA performance
# with 1k support size on 15 held-out sessions

set -e  # Exit on error

# Configuration
NUM_BASES=100
TRAIN_SIZE=5000
BASIS_RESIDUAL_RANK=16
GPU_0=0
GPU_1=1
TTA_GPU=0
TTA_EPOCHS=7001
TTA_SUPPORT_SIZE=1000
MAX_ADAPT_SESSIONS=15

# Session counts to test
SESSION_COUNTS=(5 12 15 20 25)

# Training strategies to test
STRATEGIES=("inner")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Session Count Regression Analysis${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Configuration:"
echo "  Bases: ${NUM_BASES}"
echo "  Training Size: ${TRAIN_SIZE}"
echo "  Basis Residual Rank: ${BASIS_RESIDUAL_RANK}"
echo "  Training GPUs: ${GPU_0}, ${GPU_1} (parallel)"
echo "  TTA GPU: ${TTA_GPU}"
echo ""
echo "Training:"
echo "  Training: 5, 12, 15, 20 sessions + 25-shuffle (ts${TRAIN_SIZE})"
echo "  Already trained: 25-session ts${TRAIN_SIZE}"
echo "  Total models to train: 5"
echo ""
echo "TTA Configuration:"
echo "  TTA Support Size: ${TTA_SUPPORT_SIZE}"
echo "  Max Adapt Sessions: ${MAX_ADAPT_SESSIONS}"
echo "  Total models for TTA: 6"
echo ""

# Ask for confirmation
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create results directory
RESULTS_DIR="session_count_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Results will be saved to: ${RESULTS_DIR}${NC}"
echo ""

# Array to store model paths for TTA testing
declare -a MODEL_PATHS

# Function to train a model
train_model() {
    local num_bases=$1
    local session_count=$2
    local gpu_id=$3
    local coadapt_flag=$4
    local basis_residual_rank=$5
    local train_size=$6
    local shuffle=$7
    local strategy=$8
    local results_dir=$9

    local shuffle_suffix=""
    if [ "$shuffle" = "true" ]; then
        shuffle_suffix="_shuffle"
    fi

    local model_name="${num_bases}_${session_count}_rr${basis_residual_rank}_${strategy}_ts${train_size}${shuffle_suffix}"
    local model_dir="test/${model_name}"

    echo -e "${YELLOW}[GPU ${gpu_id}] Training: ${model_name}${NC}"

    if python tma_standalone.py ${num_bases} ${session_count} ${gpu_id} ${coadapt_flag} ${basis_residual_rank} ${train_size} ${shuffle} > "${results_dir}/training_${model_name}.log" 2>&1; then
        echo -e "${GREEN}[GPU ${gpu_id}] ✓ ${model_name} complete${NC}"
        echo "${model_name}:${model_dir}" >> "${results_dir}/model_paths.txt"
    else
        echo -e "${RED}[GPU ${gpu_id}] ✗ ${model_name} failed${NC}"
    fi
}

# Phase 1: Train models in parallel
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Phase 1: Training Models (Parallel)${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Define training jobs with GPU assignments (balanced by training time)
# Format: "gpu_id session_count strategy shuffle"
# GPU 0: 20 + 5 = 25 session-units
# GPU 1: 25 + 15 + 12 = 52 session-units (unbalanced but 25-shuffle is needed)
declare -a GPU0_JOBS=(
    "${GPU_0} 20 inner false"
    "${GPU_0} 5 inner false"
)

declare -a GPU1_JOBS=(
    "${GPU_1} 25 inner true"
    "${GPU_1} 15 inner false"
    "${GPU_1} 12 inner false"
)

# Function to run jobs sequentially on a GPU
run_gpu_jobs() {
    local jobs_array_name=$1
    local -n jobs=$jobs_array_name

    for job in "${jobs[@]}"; do
        read -r gpu_id session_count strategy shuffle <<< "$job"

        if [ "$strategy" = "coadapt" ]; then
            coadapt_flag="true"
        else
            coadapt_flag="false"
        fi

        train_model ${NUM_BASES} ${session_count} ${gpu_id} ${coadapt_flag} ${BASIS_RESIDUAL_RANK} ${TRAIN_SIZE} ${shuffle} ${strategy} ${RESULTS_DIR}
    done
}

# Start GPU jobs in parallel (each GPU runs its jobs sequentially)
echo "Starting GPU 0 jobs (sequential)..."
run_gpu_jobs GPU0_JOBS &
GPU0_PID=$!

echo "Starting GPU 1 jobs (sequential)..."
run_gpu_jobs GPU1_JOBS &
GPU1_PID=$!

# Wait for both GPUs to complete all their jobs
echo ""
echo "Waiting for all training jobs to complete..."
echo "  GPU 0 (PID: $GPU0_PID): 20, 5 sessions"
echo "  GPU 1 (PID: $GPU1_PID): 25-shuffle, 15, 12 sessions"
wait $GPU0_PID $GPU1_PID

# Add newly trained model paths
echo ""
echo "Loading newly trained model paths..."
if [ -f "${RESULTS_DIR}/model_paths.txt" ]; then
    while IFS= read -r line; do
        MODEL_PATHS+=("$line")
    done < "${RESULTS_DIR}/model_paths.txt"
    echo "Loaded ${#MODEL_PATHS[@]} newly trained models"
else
    echo -e "${YELLOW}No new model paths file found (this is OK if resuming)${NC}"
fi

# Add already-trained model (25 sessions ts5000)
echo "Adding already-trained model paths..."
PRETRAINED_MODELS=(
    "100_25_rr16_inner_ts5000:test/100_25_rr16_inner_ts5000"
)

for model_path in "${PRETRAINED_MODELS[@]}"; do
    MODEL_PATHS+=("$model_path")
done

echo "Total models for TTA: ${#MODEL_PATHS[@]}"
echo ""

# Phase 2: Run TTA evaluation on all models
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Phase 2: TTA Evaluation${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Build model-paths argument
MODEL_PATHS_ARG=""
for model_path in "${MODEL_PATHS[@]}"; do
    MODEL_PATHS_ARG="${MODEL_PATHS_ARG} --model-paths ${model_path}"
done

echo "Running TTA evaluation on all models..."
echo "  Support Size: ${TTA_SUPPORT_SIZE}"
echo "  Max Adapt Sessions: ${MAX_ADAPT_SESSIONS}"
echo "  TTA Epochs: ${TTA_EPOCHS}"
echo ""

# Run TTA testing
TTA_CMD="python tta_testing.py \
    --cuda-device ${TTA_GPU} \
    --support-sizes ${TTA_SUPPORT_SIZE} \
    --max-adapt-sessions ${MAX_ADAPT_SESSIONS} \
    --tta-epochs ${TTA_EPOCHS} \
    --train-size ${TRAIN_SIZE} \
    --output-dir ${RESULTS_DIR} \
    ${MODEL_PATHS_ARG}"

echo "Command: ${TTA_CMD}"
echo ""

if eval ${TTA_CMD} > "${RESULTS_DIR}/tta_evaluation.log" 2>&1; then
    echo -e "${GREEN}✓ TTA evaluation complete${NC}"
else
    echo -e "${RED}✗ TTA evaluation failed (check ${RESULTS_DIR}/tta_evaluation.log)${NC}"
    exit 1
fi

# Phase 3: Generate summary
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Phase 3: Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Find the most recent JSON results file
RESULTS_JSON=$(ls -t ${RESULTS_DIR}/tta_support_*.json | head -1)

if [ -z "$RESULTS_JSON" ]; then
    echo -e "${RED}No results JSON file found${NC}"
    exit 1
fi

echo "Results saved to: ${RESULTS_JSON}"
echo ""

# Extract and display summary using Python
python3 << EOF
import json
import sys

try:
    with open('${RESULTS_JSON}', 'r') as f:
        results = json.load(f)

    print("Session Count vs TTA Performance Summary")
    print("=" * 80)
    print(f"{'Model':<40} {'Strategy':<15} {'Sessions':<10} {'R²':<10}")
    print("-" * 80)

    # Organize results by session count
    for run in sorted(results['runs'], key=lambda x: (x['model'], x['strategy'])):
        model = run['model']
        strategy = run['strategy']
        r2 = run['r2']

        # Extract session count from model name
        try:
            parts = model.split('_')
            if len(parts) >= 2:
                session_count = parts[1]
            else:
                session_count = 'N/A'
        except:
            session_count = 'N/A'

        print(f"{model:<40} {strategy:<15} {session_count:<10} {r2:<10.4f}")

    print("=" * 80)
    print("")
    print("To generate plots, run:")
    print(f"  python plot_tta_results.py ${RESULTS_JSON}")

except Exception as e:
    print(f"Error reading results: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Analysis Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Results directory: ${RESULTS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review results JSON: ${RESULTS_JSON}"
echo "  2. Generate plots: python plot_tta_results.py ${RESULTS_JSON}"
echo "  3. Check training logs in: ${RESULTS_DIR}/"
