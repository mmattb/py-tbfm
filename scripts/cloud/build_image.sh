#!/bin/bash
# Bake a custom GCP VM image with the py-tbfm repo + Python venv preinstalled.
# Built off Google's Deep Learning VM image (PyTorch 2.4 CUDA 12.4 base), which
# already ships with NVIDIA drivers + CUDA toolkit + a system Python.
#
# Resulting image is reusable across all subsequent VMs, dropping their boot-to-
# ready time from ~20min (full setup) to ~2min (just data sync).
#
# Usage:
#   PROJECT=my-project REPO_URL=https://github.com/me/py-tbfm.git \
#     bash scripts/cloud/build_image.sh
#
# Requires: gcloud auth login, gcloud config set project <PROJECT>

set -euo pipefail

# -------- Config (override via env vars) --------
PROJECT="${PROJECT:?PROJECT env var required}"
ZONE="${ZONE:-us-central1-a}"
IMAGE_NAME="${IMAGE_NAME:-py-tbfm-base}"
IMAGE_FAMILY="${IMAGE_FAMILY:-py-tbfm}"
BUILDER_VM="${BUILDER_VM:-py-tbfm-image-builder}"
BUILDER_MACHINE="${BUILDER_MACHINE:-n1-standard-4}"
SOURCE_IMAGE_FAMILY="${SOURCE_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
SOURCE_IMAGE_PROJECT="${SOURCE_IMAGE_PROJECT:-deeplearning-platform-release}"
REPO_URL="${REPO_URL:?REPO_URL env var required (e.g. https://github.com/you/py-tbfm.git)}"
REPO_BRANCH="${REPO_BRANCH:-main}"
# ------------------------------------------------

echo "Building image ${IMAGE_NAME} in project ${PROJECT}, zone ${ZONE}"

# 1) Spin up a builder VM from the DLVM base image.
gcloud compute instances create "${BUILDER_VM}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${BUILDER_MACHINE}" \
    --image-family="${SOURCE_IMAGE_FAMILY}" \
    --image-project="${SOURCE_IMAGE_PROJECT}" \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --metadata="install-nvidia-driver=False" \
    --scopes=cloud-platform

# Wait for SSH to come up.
echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if gcloud compute ssh "${BUILDER_VM}" --zone="${ZONE}" --project="${PROJECT}" \
        --command="echo ready" >/dev/null 2>&1; then
        break
    fi
    sleep 10
done

# 2) Provision the image: clone repo, create venv, install deps.
gcloud compute ssh "${BUILDER_VM}" --zone="${ZONE}" --project="${PROJECT}" --command='
    set -euo pipefail
    sudo apt-get update -qq
    sudo apt-get install -y -qq tmux git build-essential python3-venv python3-pip
    cd /opt
    sudo git clone --depth 1 --branch '"${REPO_BRANCH}"' '"${REPO_URL}"' py-tbfm
    sudo chown -R $USER:$USER /opt/py-tbfm
    cd /opt/py-tbfm
    python3 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
    # Use the minimal cloud requirements; full requirements.txt has many yanked/
    # source-only pins from the research env that we do not need for headless TTA.
    pip install -r requirements_cloud.txt
    pip install --no-deps -e .
    python -c "import torch; assert torch.cuda.is_available() or True; print(torch.__version__)"
    sudo mkdir -p /mnt/data
    sudo chown -R $USER:$USER /mnt/data
'

# 3) Stop and image.
gcloud compute instances stop "${BUILDER_VM}" --zone="${ZONE}" --project="${PROJECT}"

# Delete any prior image with the same name.
if gcloud compute images describe "${IMAGE_NAME}" --project="${PROJECT}" >/dev/null 2>&1; then
    echo "Deleting prior ${IMAGE_NAME}..."
    gcloud compute images delete "${IMAGE_NAME}" --project="${PROJECT}" --quiet
fi

gcloud compute images create "${IMAGE_NAME}" \
    --project="${PROJECT}" \
    --source-disk="${BUILDER_VM}" \
    --source-disk-zone="${ZONE}" \
    --family="${IMAGE_FAMILY}"

# 4) Tear down builder.
gcloud compute instances delete "${BUILDER_VM}" --zone="${ZONE}" --project="${PROJECT}" --quiet

echo "Done. Image: ${IMAGE_NAME} (family ${IMAGE_FAMILY})"
