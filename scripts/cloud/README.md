# Cloud runbook (GCP)

End-to-end recipe for running the cross-animal and calibration-draw experiments
on GCP spot VMs.

## Prereqs

- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- A GCP project with billing enabled and Compute Engine + Cloud Storage APIs on.
- A2 (A100) quota in your chosen region. Default region in these scripts is
  `us-central1-a`. Request quota in the GCP console under IAM → Quotas.
- A budget alert set on the project ($100 / $250 / $500 thresholds recommended).
- Code pushed to a Git remote (the image build pulls from `REPO_URL`).

Set these once per shell:

```bash
export PROJECT=<your-gcp-project>
export BUCKET=<unique-bucket-name>          # e.g. py-tbfm-<your-name>
export REPO_URL=<your-git-https-url>        # used by build_image.sh
export ZONE=us-central1-a                   # optional, this is the default
```

## One-time setup

### 1. Upload data to GCS (~50 min on 1 Gbps; $0 ingress)

```bash
bash scripts/cloud/sync_data_to_gcs.sh --dry-run    # confirm scope
bash scripts/cloud/sync_data_to_gcs.sh              # real upload
```

Storage: ~$7.58/mo for 379 GB at standard tier.

### 2. Build the custom VM image (~20 min, ~$0.10)

```bash
bash scripts/cloud/build_image.sh
```

This produces image `py-tbfm-base` in your project. Re-run any time you want to
bake a new commit into the base image; it will replace the prior version.

## Running an experiment

Same pattern for both experiments. Pick a VM size based on parallelism need:

- Cross-animal: 2 GPUs (one per fold during training).
- Calibration draws: 8 GPUs (fans 20 draws across 8 GPUs in 3 waves, ~9h).

### 3. Launch a spot VM

```bash
# 2-GPU box for cross-animal
bash scripts/cloud/launch_vm.sh xanimal 2

# 8-GPU box for calibration draws
bash scripts/cloud/launch_vm.sh caldraw 8
```

The VM boots from `py-tbfm-base`, then its startup script rsyncs data from GCS
into `/mnt/data`. Watch progress:

```bash
gcloud compute ssh xanimal --zone=$ZONE --project=$PROJECT \
    --command='tail -f /var/log/tbfm-startup.log'
```

When `~/.tbfm_ready` exists on the VM, it's ready.

### 4. Run the experiment inside tmux

```bash
gcloud compute ssh xanimal --zone=$ZONE --project=$PROJECT
# then on the VM:
cd /opt/py-tbfm
source .venv/bin/activate
export TBFM_DATA_DIR=/mnt/data
tmux new -s run

# cross-animal:
bash scripts/train_cross_animal.sh cross_animal_$(date +%Y%m%d)
bash scripts/tta_cross_animal.sh cross_animal_$(date +%Y%m%d)

# calibration draws (one session at a time, repeat for each):
bash scripts/calibration_draw_sweep.sh \
    /opt/py-tbfm/random_folds_20260101_211200/fold0 \
    MonkeyJ_20160426_Session1_S1 \
    caldraw_J1 8
```

Detach with `Ctrl-b d`. The job continues; reattach later with `tmux attach -t run`.

### 5. Tear down when done

```bash
bash scripts/cloud/teardown_vm.sh xanimal cross_animal_20260505
bash scripts/cloud/teardown_vm.sh caldraw caldraw_J1
```

This rsyncs results to `gs://$BUCKET/results/<vm-name>/` and deletes the VM.

### 6. Pull results to local

```bash
gsutil -m rsync -r gs://$BUCKET/results/xanimal/ ./xanimal_results/
```

## Spot eviction

A2 spot VMs can be preempted. Both modified scripts (`tta_cross_animal.sh`,
`calibration_draw_sweep.sh`) wrap their Python invocations in
`scripts/cloud/retry_on_preemption.sh`, which retries up to 3× on non-zero
exit. Since TTA runs are ~3.5h and write all output at the end, a preempted
job restarts from scratch — fine in expectation.

If a VM itself is preempted (terminated), `--instance-termination-action=DELETE`
removes it. Re-run `launch_vm.sh` with the same name to provision a fresh one;
data re-syncs automatically.

## Cost guardrails

- Spot pricing assumed: `a2-highgpu-2g` ≈ $1.76/hr, `a2-highgpu-8g` ≈ $7.04/hr.
  Check current prices: `gcloud compute machine-types describe a2-highgpu-8g --zone=$ZONE`.
- Always run `teardown_vm.sh` after a session — an idle 8x A100 box is ~$170/day.
- Set a project budget alert in the GCP console.
- Cap A100 quota in your region (e.g., 8 GPUs) to prevent accidental fan-out.
