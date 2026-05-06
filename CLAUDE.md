# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running scripts

Always run long-running scripts (training, TTA, sweeps) inside a tmux session with a descriptive name — never as a background task or foreground process.

```bash
tmux new-session -d -s <descriptive_name> -c /home/danmuir/GitHub/py-tbfm
tmux send-keys -t <descriptive_name> "python ..." Enter
# Attach with: tmux attach -t <descriptive_name>
```

## Key entry points

- **`tma_standalone.py`** — main training script. Positional args: `num_bases num_sessions gpu_id coadapt basis_residual_rank train_size shuffle`. Overrides `lambda_ae_recon=0.03` and `epochs=12001`. Data is read from `$TBFM_DATA_DIR` (default `/var/data/opto-coproc/`).
- **`tta_testing.py`** — TTA evaluation across models, support sizes, and strategies. Supports multi-GPU via `--use-multi-gpu --gpu-ids 0 1`.
- **`scripts/train_random_folds.sh`** — trains 20 random folds of 20 sessions each (2 GPUs in parallel).
- **`scripts/tta_random_folds.sh <folds_dir>`** — runs TTA on all folds from a training run.

Standard TTA flags used across experiments:
```
--unfreeze-bases --progressive-unfreezing-threshold 0 --max-adapt-sessions 20 --tta-epochs 7001
```

Output directories follow `<folds_dir>/tta_results_<support_size>_<timestamp>/fold<i>`.

## Architecture

The model is a **Temporal Basis Function Model (TBFM)** extended to the multi-session setting. A forward pass flows through:

1. **Normalizers** (`tbfm/normalizers.py`) — per-session z-score normalization of neural activity (runway).
2. **Autoencoder** (`tbfm/ae.py`) — encodes normalized runway into a low-dimensional latent. `LinearChannelAE` (default) or `TwoStageAffineAE`. Session-specific instances are managed by `SessionDispatcherLinearAE`.
3. **TBFM** (`tbfm/tbfm.py`) — predicts future latent activity using a learned set of temporal basis functions. Basis generator is an MLP conditioned on stimulus covariates and session embeddings. Basis weights are predicted from the runway latent. Output: weighted sum of bases + skip connection from last latent timestep.
4. **AE decode** — maps latent forecast back to neural activity space.

This full pipeline is wrapped in `TBFMMultisession` (`tbfm/_multisession_module.py`). Session-specific TBFM instances are managed by `SessionDispatcherTBFM`.

**Meta-learning (MAML-style, default):** Each training episode splits session data into support and query sets. An inner loop (`meta.inner_update_stopgrad`) adapts per-session stimulus embeddings (`embeddings_stim`) on the support set. The outer loop updates shared TBFM and AE parameters on the query set using the adapted embeddings.

**Co-adaptation mode** (`meta.training.coadapt=true`): skips the inner loop; embeddings and AE are jointly optimized in a single loop.

**Test-time adaptation (TTA):** `multisession.test_time_adaptation()` mirrors training — AE is warm-started via PCA, then adapted using the same inner/outer loop structure. The outer loss includes both supervised prediction MSE and AE reconstruction MSE (weighted by `cfg.ae.training.lambda_ae_recon`). Progressive unfreezing optionally adds TBFM basis/weight updates at high support sizes.

## Config system

Hydra configs live in `conf/`. The hierarchy is `config.yaml` → `tbfm/default.yaml`, `ae/default.yaml`, `meta/default.yaml`, `normalizers/default.yaml`. Key overrides in `tma_standalone.py`:

- `latent_dim`: AE/TBFM latent dimension (default 85 in standalone, 80 in conf)
- `cfg.ae.training.lambda_ae_recon`: reconstruction loss weight (set to `0.03` in standalone, `0.01` in conf)
- `cfg.tbfm.module.num_bases`: number of temporal basis functions

## Saved model artifacts

Each trained model directory contains:
- `model.torch` / `model_nf_1.torch` — TBFM weights
- `hisi.torch` / `hisi_nf_1.torch` — held-in session IDs
- `hyperparameters.torch` — dict with `latent_dim`, `num_bases`, `basis_residual_rank`, `residual_mlp_hidden`, `embed_dim_stim`, `coadapt`
- `es.torch` — rest embeddings; `r.torch` — results

TTA adapted model dirs additionally contain `embeddings_stim.torch` and `metadata.torch`.
