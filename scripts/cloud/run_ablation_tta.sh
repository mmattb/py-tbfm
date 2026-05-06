#!/bin/bash
# Run TTA for a single ablation on a GCP VM. Pulls the trained ablation model
# from GCS, then invokes scripts/tta_ablations.sh restricted to that ablation.
# Wraps the TTA call in retry_on_preemption.sh so spot evictions get one
# automatic retry before requiring manual relaunch.
#
# Usage (on a GCP VM):
#   BUCKET=my-bucket bash scripts/cloud/run_ablation_tta.sh <ablation> <support_size>
#
# Examples:
#   bash scripts/cloud/run_ablation_tta.sh no_ae_recon 5000
#   bash scripts/cloud/run_ablation_tta.sh with_ortho 5000

set -euo pipefail

ABL="${1:?usage: $0 <ablation> <support_size>}"
SUPPORT="${2:?usage: $0 <ablation> <support_size>}"
BUCKET="${BUCKET:?BUCKET env var required}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/ablations_remote}"

# Pull the trained ablation model from GCS.
mkdir -p "${WORK_DIR}/${ABL}"
echo "Syncing gs://${BUCKET}/ablations/${ABL}/ -> ${WORK_DIR}/${ABL}/"
gsutil -m rsync -r "gs://${BUCKET}/ablations/${ABL}/" "${WORK_DIR}/${ABL}/"

cd "${REPO_ROOT}"
export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"

echo "Running TTA: ablation=${ABL}, support=${SUPPORT}, work_dir=${WORK_DIR}"
SUPPORT_SIZE="${SUPPORT}" \
    ABLATION_NAMES=("${ABL}") \
    "${RETRY}" bash scripts/tta_ablations.sh "${WORK_DIR}"

echo "Done. Result: ${WORK_DIR}/tta_${ABL}_${SUPPORT}/"
