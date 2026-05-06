#!/bin/bash
# Within-session calibration draw sensitivity.
#
# For a fixed held-out session, draws K random calibration sets of size N,
# runs TTA each time, and records per-draw test R². Isolates variance from
# MAML's inner-loop sampling mechanism rather than session-level variability.
#
# Draws are fanned out across all visible GPUs (or NUM_GPUS if specified).
# Each TTA run is wrapped in scripts/cloud/retry_on_preemption.sh so spot
# preemption mid-draw simply restarts that draw.
#
# Usage: bash scripts/calibration_draw_sweep.sh <model_dir> <session_id> [output_dir] [num_gpus]
#
# Example (local 2-GPU box):
#   bash scripts/calibration_draw_sweep.sh \
#     random_folds_20260101_211200/fold0 \
#     MonkeyJ_20160426_Session1_S1
#
# Example (cloud 8-GPU VM):
#   bash scripts/calibration_draw_sweep.sh \
#     /opt/py-tbfm/random_folds_20260101_211200/fold0 \
#     MonkeyJ_20160426_Session1_S1 \
#     /opt/py-tbfm/calibration_draw_J1 8

set -u

MODEL_DIR="${1:-}"
SESSION_ID="${2:-}"
OUTPUT_BASE="${3:-calibration_draw_$(date +%Y%m%d_%H%M%S)}"

# Auto-detect GPU count if not specified.
if [ -n "${4:-}" ]; then
    NUM_GPUS="${4}"
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${NUM_GPUS}" -eq 0 ]; then NUM_GPUS=1; fi
fi

if [ -z "${MODEL_DIR}" ] || [ -z "${SESSION_ID}" ]; then
    echo "Usage: $0 <model_dir> <session_id> [output_dir] [num_gpus]"
    exit 1
fi

MODEL_DIR=$(realpath "${MODEL_DIR}")
mkdir -p "${OUTPUT_BASE}"

K=20           # number of random draws
SUPPORT_SIZE=2500
TTA_EPOCHS=7001

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"

echo "Calibration draw sensitivity sweep"
echo "  Model:        ${MODEL_DIR}"
echo "  Session:      ${SESSION_ID}"
echo "  Draws (K):    ${K}"
echo "  Support size: ${SUPPORT_SIZE}"
echo "  Output:       ${OUTPUT_BASE}"
echo "  GPUs:         ${NUM_GPUS}"
echo ""

# GPU pool: track one PID per GPU; on each iteration, dispatch draw to next free GPU.
declare -A GPU_PIDS
declare -A GPU_SEEDS
for g in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_PIDS[$g]=""
done

run_draw() {
    local seed=$1
    local gpu=$2
    local out_dir="${OUTPUT_BASE}/draw_${seed}"
    local log="${OUTPUT_BASE}/draw_${seed}.log"

    bash "${RETRY}" python -u tta_testing.py \
        --model-paths "model:${MODEL_DIR}" \
        --adapt-session "${SESSION_ID}" \
        --support-sizes ${SUPPORT_SIZE} \
        --tta-epochs ${TTA_EPOCHS} \
        --unfreeze-bases \
        --progressive-unfreezing-threshold 0 \
        --random-support \
        --support-seed ${seed} \
        --cuda-device ${gpu} \
        --output-dir "${out_dir}" \
        > "${log}" 2>&1
}

wait_on_gpu() {
    local gpu=$1
    local pid="${GPU_PIDS[$gpu]}"
    local seed="${GPU_SEEDS[$gpu]}"
    if [ -z "${pid}" ]; then return 0; fi
    wait "${pid}"
    local rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "  Draw ${seed} (gpu ${gpu}) FAILED (exit ${rc}). Log: ${OUTPUT_BASE}/draw_${seed}.log"
    else
        echo "  Draw ${seed} (gpu ${gpu}) done."
    fi
    GPU_PIDS[$gpu]=""
}

for SEED in $(seq 0 $((K - 1))); do
    GPU_ID=$((SEED % NUM_GPUS))
    # If this GPU is busy with a prior draw, wait for it.
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        wait_on_gpu ${GPU_ID}
    fi
    echo "Dispatching draw ${SEED}/${K} to GPU ${GPU_ID}..."
    run_draw ${SEED} ${GPU_ID} &
    GPU_PIDS[$GPU_ID]=$!
    GPU_SEEDS[$GPU_ID]=${SEED}
done

# Drain remaining GPUs.
for g in $(seq 0 $((NUM_GPUS - 1))); do
    wait_on_gpu ${g}
done

echo ""
echo "All ${K} draws complete. Analyzing variance..."

python -u - <<EOF
import json, csv
import numpy as np
from pathlib import Path

output_base = Path("${OUTPUT_BASE}")
session_id = "${SESSION_ID}"
k = ${K}
support_size = ${SUPPORT_SIZE}

r2s = []
for seed in range(k):
    draw_dir = output_base / f"draw_{seed}"
    for fname in ["per_session_results.csv", "tta_runs.csv"]:
        csv_path = draw_dir / fname
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    sid = row.get("session_id") or row.get("session", "")
                    if session_id in sid:
                        try:
                            r2s.append(float(row.get("r2") or row.get("final_r2") or row.get("test_r2", "")))
                        except (ValueError, TypeError):
                            pass
            break

if r2s:
    arr = np.array(r2s)
    print(f"\nCalibration draw sensitivity for {session_id}")
    print(f"  Support size: {support_size}, K={k} draws (collected={len(r2s)})")
    print(f"  R²: {arr.mean():.4f} ± {arr.std():.4f}  (min={arr.min():.4f}, max={arr.max():.4f})")
    summary = {
        "session_id": session_id,
        "support_size": support_size,
        "k": k,
        "n_collected": len(r2s),
        "r2_mean": float(arr.mean()),
        "r2_std": float(arr.std()),
        "r2_min": float(arr.min()),
        "r2_max": float(arr.max()),
        "r2_per_draw": r2s,
    }
    out_path = output_base / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {out_path}")
else:
    print(f"No R² values found — check draw logs in {output_base}")
EOF
