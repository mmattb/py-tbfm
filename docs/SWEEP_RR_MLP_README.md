# Basis Residual Rank, MLP Hidden Dimension, and Stim Embedding Dimension Sweep

## Overview

This sweep evaluates the impact of three key hyperparameters on TTA (Test-Time Adaptation) performance:
1. **Basis Residual Rank (rr)**: The rank of the residual subspace
2. **MLP Hidden Dimension**: The hidden layer size of the residual MLP
3. **Stim Embedding Dimension (eds)**: The dimensionality of the stimulus embedding

## Fixed Parameters

- **Latent Dimension**: 96
- **Number of Bases**: 100
- **Number of Training Sessions**: 25
- **Batch Size**: 500 per session (12,500 total batch size)
- **Training Data**: All available training data (5000 samples)
- **Training Strategy**: Inner loop (MAML-style, not co-adaptation)
- **Data Shuffling**: Disabled

## Sweep Parameters

- **RR Values**: [4, 8, 16, 32, 64]
- **MLP Hidden Values**: [8, 16, 32, 64, 128]
- **Stim Embed Dim Values**: [5, 10, 15, 20, 30]
- **Total Configurations**: 5 × 5 × 5 = 125 models

## Evaluation Metric

All models are evaluated using:
- **TTA Setup**: 5 held-out sessions
- **Support Size**: 500 samples
- **TTA Epochs**: 7001
- **Inner Steps**: 20

## Usage

### Run the Complete Sweep

```bash
./sweep_rr_mlp.sh
```

This will:
1. **Phase 1**: Train all 25 model configurations in parallel across 2 GPUs
2. **Phase 2**: Evaluate all models using multi-GPU TTA

### Customize GPUs

Edit the script to change which GPUs to use:
```bash
GPU_IDS=(0 1)  # Change to your desired GPUs, e.g., (0 1 2 3) for 4 GPUs
```

The script will:
- Train models in parallel, distributing them across available GPUs
- Use all specified GPUs for TTA evaluation via `--use-multi-gpu`

### Customize Sweep Parameters

Edit the arrays in the script:
```bash
RR_VALUES=(4 8 16 32 64)                    # Modify these values
MLP_HIDDEN_VALUES=(8 16 32 64 128)          # Modify these values
EMBED_DIM_STIM_VALUES=(5 10 15 20 30)       # Modify these values
```

## Output Structure

```
sweep_rr_mlp_YYYYMMDD_HHMMSS/
├── models/                          # Trained models
│   ├── 100_25_rr4_inner_ts5000_shuffle_ld96_bs12500_mlp8_eds5/
│   ├── 100_25_rr4_inner_ts5000_shuffle_ld96_bs12500_mlp8_eds10/
│   └── ...
├── tta_results/                     # TTA evaluation results
│   ├── tta_support_YYYYMMDD_HHMMSS.json
│   └── tta_support_YYYYMMDD_HHMMSS.png
└── model_paths.txt                  # Model paths for TTA

```

## Model Naming Convention

Models are named: `{num_bases}_{num_sessions}_rr{rr}_inner_ts{train_size}_shuffle_ld{latent_dim}_bs{batch_size}_mlp{mlp_hidden}_eds{embed_dim_stim}`

Example: `100_25_rr16_inner_ts5000_shuffle_ld96_bs12500_mlp32_eds15`
- 100 bases
- 25 training sessions
- Residual rank 16
- Inner loop training
- 5000 training samples (all data)
- Shuffle enabled
- Latent dimension 96
- Batch size 12,500 (500 per session × 25 sessions)
- MLP hidden dimension 32
- Stim embedding dimension 15

## Expected Runtime

With 2 GPUs running in parallel:
- **Training**: ~30-60 minutes per model
- **Total Training Time**: ~31-62 hours (125 models parallelized across 2 GPUs)
- **TTA Evaluation**: ~5-10 hours (multi-GPU parallelized)

Total sweep time: **~36-72 hours** (1.5-3 days)

## Analyzing Results

After the sweep completes, analyze the TTA results:

```bash
python plot_tta_results.py sweep_rr_mlp_YYYYMMDD_HHMMSS/tta_results
```

The JSON results file contains:
- R² scores for each configuration
- Support size performance curves
- Metadata about the evaluation

## Modifications

### Changes to tma_standalone.py

The script was modified to accept additional command-line arguments:
1. `--latent-dim`: Controls `cfg.latent_dim` (autoencoder latent dimension)
2. `--batch-size-per-session`: Controls batch size for data loading
3. `--residual-mlp-hidden`: Controls `cfg.meta.residual_mlp_hidden` (MLP hidden dimension)
4. `--embed-dim-stim`: Controls `cfg.tbfm.module.embed_dim_stim` (stimulus embedding dimension)

Note: `train_size` parameter controls how much training data to use (5000 = all data), while `batch_size_per_session` controls the batch size for loading data from each session.

These parameters are now included in the output directory naming for easy identification.

### Changes to Sweep Script

The sweep script:
- **3D Parameter Sweep**: Sweeps over RR, MLP hidden dimension, AND stim embedding dimension
- **Parallel Training**: Distributes models across 2 GPUs using background jobs
- **Round-robin GPU Assignment**: Automatically assigns jobs to available GPUs
- **Job Tracking**: Monitors GPU availability and waits for completion
- **Thread-safe Logging**: Uses file locking for concurrent model path recording
- **Multi-GPU TTA**: Leverages `tta_testing.py --use-multi-gpu` for parallel evaluation
- Organizes output into timestamped directories

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
- Reduce `BATCH_SIZE_PER_SESSION` (e.g., 250 instead of 500)
- Reduce `NUM_SESSIONS` (e.g., 20 instead of 25)
- Reduce number of sessions in TTA evaluation
- Use smaller models in the sweep ranges

### Slow Training

To speed up:
- Add more GPUs to `GPU_IDS` array for more parallelism
- Reduce `cfg.training.epochs` in `tma_standalone.py`
- Remove some configurations from the sweep

### GPU Conflicts

If jobs fail due to GPU conflicts:
- Ensure no other processes are using the GPUs
- Check GPU memory with `nvidia-smi`
- Reduce number of parallel jobs by using fewer GPUs

### TTA Errors

If TTA fails:
- Check that all models trained successfully
- Verify model paths in `model_paths.txt`
- Check GPU memory during TTA evaluation
