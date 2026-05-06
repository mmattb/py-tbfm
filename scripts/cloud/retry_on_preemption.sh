#!/bin/bash
# Retry a command up to 3 times on non-zero exit, with a 60s wait between attempts.
# Intended to absorb GCP spot-VM preemptions for short (~3-4h) jobs that write
# all output at the end and can safely restart from scratch.
#
# Usage: retry_on_preemption.sh <cmd> [args...]

set -u

MAX_ATTEMPTS=3
WAIT_SEC=60

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cmd> [args...]" >&2
    exit 2
fi

attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[retry_on_preemption] attempt ${attempt}/${MAX_ATTEMPTS}: $*"
    "$@"
    rc=$?
    if [ $rc -eq 0 ]; then
        exit 0
    fi
    echo "[retry_on_preemption] attempt ${attempt} exited ${rc}"
    if [ $attempt -lt $MAX_ATTEMPTS ]; then
        echo "[retry_on_preemption] sleeping ${WAIT_SEC}s before retry"
        sleep $WAIT_SEC
    fi
    attempt=$((attempt + 1))
done

echo "[retry_on_preemption] all ${MAX_ATTEMPTS} attempts failed; giving up" >&2
exit $rc
