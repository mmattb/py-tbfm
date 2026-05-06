#!/bin/bash
# Launch a GCP spot VM from the py-tbfm-base image, with a startup script that
# rsyncs data from GCS into the local SSD and writes a ~/.tbfm_ready sentinel.
# Does NOT auto-run the experiment — SSH in and launch inside tmux.
#
# Usage:
#   PROJECT=my-project BUCKET=my-bucket bash scripts/cloud/launch_vm.sh \
#     <vm-name> <gpu-count>
#
# Examples:
#   bash scripts/cloud/launch_vm.sh xanimal 2     # 2x A100 for cross-animal
#   bash scripts/cloud/launch_vm.sh caldraw 8     # 8x A100 for calibration draws
#
# Watch readiness: gcloud compute ssh <vm-name> -- 'tail -f /var/log/tbfm-startup.log'
# When ~/.tbfm_ready exists, data is synced and the VM is ready.

set -euo pipefail

PROJECT="${PROJECT:?PROJECT env var required}"
BUCKET="${BUCKET:?BUCKET env var required}"
ZONE="${ZONE:-us-central1-a}"
IMAGE_FAMILY="${IMAGE_FAMILY:-py-tbfm}"
BOOT_DISK_GB="${BOOT_DISK_GB:-200}"

VM_NAME="${1:?usage: $0 <vm-name> <gpu-count>}"
GPU_COUNT="${2:?usage: $0 <vm-name> <gpu-count>}"

case "${GPU_COUNT}" in
    1) MACHINE_TYPE="a2-highgpu-1g" ;;
    2) MACHINE_TYPE="a2-highgpu-2g" ;;
    4) MACHINE_TYPE="a2-highgpu-4g" ;;
    8) MACHINE_TYPE="a2-highgpu-8g" ;;
    *) echo "ERROR: gpu-count must be 1, 2, 4, or 8 (got ${GPU_COUNT})" >&2; exit 1 ;;
esac

echo "Launching ${VM_NAME}: ${MACHINE_TYPE} (${GPU_COUNT}x A100), spot, in ${ZONE}"

# Startup script runs on VM boot. Logs to /var/log/tbfm-startup.log.
# Mounts local SSD at /mnt/data, rsyncs data from GCS, writes ready sentinel.
STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
exec > /var/log/tbfm-startup.log 2>&1
set -euxo pipefail

# Format and mount local SSD if present, else use boot disk.
if [ -e /dev/nvme0n1 ] && ! mountpoint -q /mnt/data; then
    mkfs.ext4 -F /dev/nvme0n1 || true
    mkdir -p /mnt/data
    mount -o discard,defaults /dev/nvme0n1 /mnt/data
fi
mkdir -p /mnt/data
chmod 777 /mnt/data

# Pull data from GCS (parallel, resumable).
sudo -u \$(logname 2>/dev/null || echo \$USER) \
    gsutil -m rsync -r gs://${BUCKET}/data/ /mnt/data/

# Pull latest repo (image has a snapshot; refresh in case of new commits).
cd /opt/py-tbfm
git fetch --depth 1 origin "\$(git rev-parse --abbrev-ref HEAD)" || true
git pull --ff-only || true

# Mark ready.
touch /home/\$(ls /home | head -1)/.tbfm_ready
echo "READY at \$(date)"
EOF
)

# Build the gcloud invocation.
# --provisioning-model=SPOT + --instance-termination-action=DELETE for spot pricing.
# --local-ssd is included in a2 machine prices.
gcloud compute instances create "${VM_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${PROJECT}" \
    --boot-disk-size="${BOOT_DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --local-ssd=interface=NVME \
    --metadata="install-nvidia-driver=False" \
    --metadata-from-file=startup-script=<(echo "${STARTUP_SCRIPT}") \
    --scopes=cloud-platform \
    --maintenance-policy=TERMINATE

echo ""
echo "VM ${VM_NAME} created. Watch setup with:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT} \\"
echo "    --command='tail -f /var/log/tbfm-startup.log'"
echo ""
echo "When ~/.tbfm_ready exists, attach with:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
echo ""
echo "Then inside the VM:"
echo "  cd /opt/py-tbfm && source .venv/bin/activate"
echo "  export TBFM_DATA_DIR=/mnt/data"
echo "  tmux new -s run"
echo "  bash scripts/<your-experiment>.sh"
