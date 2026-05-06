#!/bin/bash
# One-time upload of /var/data/opto-coproc to a GCS bucket.
# After this runs, VMs read data via gsutil rsync from the bucket on boot.
#
# Usage:
#   BUCKET=my-bucket bash scripts/cloud/sync_data_to_gcs.sh           # real run
#   BUCKET=my-bucket bash scripts/cloud/sync_data_to_gcs.sh --dry-run # preview only
#
# Requires: gcloud auth login (with a user that can write to BUCKET).

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
SOURCE_DIR="${SOURCE_DIR:-/var/data/opto-coproc}"
DEST="gs://${BUCKET}/data/"
DRY_RUN=""

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="-n"
    echo "[dry-run] no files will be uploaded"
fi

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "ERROR: source directory not found: ${SOURCE_DIR}" >&2
    exit 1
fi

# Create bucket if it doesn't exist (idempotent).
if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
    echo "Creating bucket gs://${BUCKET}..."
    gsutil mb -l us-central1 "gs://${BUCKET}"
fi

SIZE=$(du -sh "${SOURCE_DIR}" | cut -f1)
echo "Source: ${SOURCE_DIR} (${SIZE})"
echo "Dest:   ${DEST}"
echo ""

# -m for parallel transfers, -r for recursive, rsync for delta-only re-uploads.
gsutil -m rsync -r ${DRY_RUN} "${SOURCE_DIR}" "${DEST}"

echo ""
echo "Done. Verify with: gsutil ls ${DEST} | head"
