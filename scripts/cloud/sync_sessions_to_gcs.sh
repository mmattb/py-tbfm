#!/bin/bash
# Upload only the sessions used by the fold0 ablation experiments (held-in + held-out)
# plus the top-level pickle files. ~283 GB instead of full 379 GB.
#
# Usage:
#   BUCKET=my-bucket bash scripts/cloud/sync_sessions_to_gcs.sh           # real run
#   BUCKET=my-bucket bash scripts/cloud/sync_sessions_to_gcs.sh --dry-run # preview
#
# Per-session rsync sidesteps a broken symlink in the global tree that crashes
# `gsutil rsync` at the SOURCE_DIR level.

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
SOURCE_DIR="${SOURCE_DIR:-/var/data/opto-coproc}"
DEST="gs://${BUCKET}/data/"
DRY_RUN=""

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="-n"
    echo "[dry-run] no files will be uploaded"
fi

# Held-in fold0 sessions (20)
HELD_IN=(
    MonkeyG_20150915_Session2_S1 MonkeyG_20150915_Session3_S1 MonkeyG_20150915_Session4_S1
    MonkeyG_20150916_Session4_S1 MonkeyG_20150917_Session1_S1 MonkeyG_20150917_Session2_M1
    MonkeyG_20150917_Session3_M1 MonkeyG_20150921_Session5_S1 MonkeyG_20150922_Session2_S1
    MonkeyG_20150922_Session3_S1 MonkeyJ_20160426_Session3_S1 MonkeyJ_20160428_Session2_S1
    MonkeyJ_20160428_Session3_S1 MonkeyJ_20160429_Session3_S1 MonkeyJ_20160624_Session3_S1
    MonkeyJ_20160624_Session4_S1 MonkeyJ_20160625_Session4_S1 MonkeyJ_20160625_Session5_S1
    MonkeyJ_20160627_Session2_S1 MonkeyJ_20160630_Session1_S1
)

# Held-out fold0 sessions (20)
HELD_OUT=(
    MonkeyG_20150914_Session1_S1 MonkeyG_20150914_Session3_S1 MonkeyG_20150915_Session5_S1
    MonkeyG_20150917_Session1_M1 MonkeyG_20150917_Session2_S1 MonkeyG_20150917_Session3_S1
    MonkeyG_20150918_Session1_M1 MonkeyG_20150918_Session1_S1 MonkeyG_20150921_Session3_S1
    MonkeyG_20150922_Session1_S1 MonkeyG_20150925_Session1_S1 MonkeyG_20150925_Session2_S1
    MonkeyJ_20160426_Session1_S1 MonkeyJ_20160426_Session2_S1 MonkeyJ_20160429_Session1_S1
    MonkeyJ_20160502_Session1_S1 MonkeyJ_20160627_Session1_S1 MonkeyJ_20160630_Session3_S1
    MonkeyJ_20160702_Session2_S1 MonkeyJ_20160702_Session4_S1
)

# Create bucket if missing.
if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
    echo "Creating bucket gs://${BUCKET}..."
    gsutil mb -l us-central1 "gs://${BUCKET}"
fi

ALL_SESSIONS=("${HELD_IN[@]}" "${HELD_OUT[@]}")

# Sessions with broken symlinks that crash gsutil's listing — staged via rsync
# into a temp dir first (rsync handles broken symlinks gracefully).
PROBLEM_SESSIONS=(MonkeyJ_20160702_Session4_S1)

# Top-level pickle files first (small, cheap).
echo "Syncing top-level files..."
gsutil -m cp ${DRY_RUN} \
    "${SOURCE_DIR}/bad_channels.pkl" \
    "${SOURCE_DIR}/electrode_positions.pkl" \
    "${DEST}" || true

is_problem_session() {
    local s="$1"
    for p in "${PROBLEM_SESSIONS[@]}"; do
        [ "$s" = "$p" ] && return 0
    done
    return 1
}

STAGING_DIR="$(mktemp -d -t gcs_staging.XXXXXX)"
trap 'rm -rf "${STAGING_DIR}"' EXIT
echo "Staging dir for problem sessions: ${STAGING_DIR}"

# Per-session rsync. Each session gets its own gsutil call so a problem in one
# doesn't abort the rest.
for sid in "${ALL_SESSIONS[@]}"; do
    SRC="${SOURCE_DIR}/${sid}"
    if [ ! -d "${SRC}" ]; then
        echo "WARNING: ${SRC} not found, skipping"
        continue
    fi

    if is_problem_session "${sid}"; then
        # Stage a clean copy via rsync — strips ALL symlinks (broken `data` link
        # plus cross-session links we don't need), keeps real subdirs only.
        STAGED="${STAGING_DIR}/${sid}"
        echo "Staging ${sid} via rsync (skipping symlinks)..."
        mkdir -p "${STAGED}"
        rsync -a --no-links "${SRC}/" "${STAGED}/"
        SYNC_SRC="${STAGED}"
    else
        SYNC_SRC="${SRC}"
    fi

    echo "Syncing ${sid}..."
    gsutil -m rsync -r ${DRY_RUN} "${SYNC_SRC}" "${DEST}${sid}/" || {
        echo "ERROR: failed to sync ${sid}, continuing"
    }
done

echo ""
echo "Done. Verify with: gsutil ls ${DEST} | head"
