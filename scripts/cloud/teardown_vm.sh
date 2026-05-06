#!/bin/bash
# Sync results from a VM back to GCS, then delete the VM.
# Run this when an experiment finishes — leaving idle a2 VMs running burns money fast.
#
# Usage:
#   PROJECT=my-project BUCKET=my-bucket bash scripts/cloud/teardown_vm.sh \
#     <vm-name> [results-subdir]
#
# Example:
#   bash scripts/cloud/teardown_vm.sh xanimal cross_animal_20260505
#
# If results-subdir is omitted, defaults to /opt/py-tbfm (everything).

set -euo pipefail

PROJECT="${PROJECT:?PROJECT env var required}"
BUCKET="${BUCKET:?BUCKET env var required}"
ZONE="${ZONE:-us-central1-a}"

VM_NAME="${1:?usage: $0 <vm-name> [results-subdir]}"
RESULTS_SUBDIR="${2:-}"

if [ -n "${RESULTS_SUBDIR}" ]; then
    REMOTE_PATH="/opt/py-tbfm/${RESULTS_SUBDIR}"
else
    # Sync all common output dirs.
    REMOTE_PATH="/opt/py-tbfm"
fi

DEST="gs://${BUCKET}/results/${VM_NAME}/"

echo "Syncing ${VM_NAME}:${REMOTE_PATH} -> ${DEST}"

# Run gsutil rsync from the VM itself — much faster than going through local box.
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT}" --command="
    set -e
    if [ -d '${REMOTE_PATH}' ]; then
        gsutil -m rsync -r \
            -x '.*\.venv/.*|.*__pycache__/.*|.*\.git/.*|.*/data/.*' \
            '${REMOTE_PATH}' '${DEST}'
    else
        echo 'WARNING: ${REMOTE_PATH} not found on VM' >&2
    fi
"

echo ""
echo "Deleting VM ${VM_NAME}..."
gcloud compute instances delete "${VM_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --quiet

echo ""
echo "Done. Pull results to local with:"
echo "  gsutil -m rsync -r ${DEST} ./<local-dir>/"
