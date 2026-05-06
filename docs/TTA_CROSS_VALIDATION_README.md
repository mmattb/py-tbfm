# TTA Cross-Validation Scripts

This document describes the test-time adaptation (TTA) scripts for the second half of cross-validation after running `train_random_folds.sh`.

## Overview

The cross-validation workflow has two parts:

1. **Training Phase** (`train_random_folds.sh`): Train 20 models, each on a random subset of 20 sessions
2. **TTA Phase** (these scripts): Run TTA on the 20 held-out sessions for each fold

## Scripts

### 1. `tta_random_folds.sh` - Batch TTA Mode

Runs TTA on all 20 held-out sessions together for each fold. Faster but provides one adapted model per fold.

**Usage:**
```bash
./tta_random_folds.sh <path_to_random_folds_results>
```

**Example:**
```bash
./tta_random_folds.sh random_folds_20260108_123456
```

**Features:**
- Processes all 20 held-out sessions together per fold
- Uses multi-GPU (GPUs 0 and 1) for parallel processing
- Saves one adapted model per (fold, strategy, support_size)
- Saves per-session R² scores in metadata
- Faster execution (~1 TTA run per fold)

**Output Structure:**
```
random_folds_TIMESTAMP/
  tta_results_TIMESTAMP/
    fold0/
      tta_support_*.json       # Results with per-session R²
      adapted_models/          # Adapted model (all sessions combined)
    fold1/
      ...
    tta_timing_log.txt         # Detailed timing information
    tta_summary.csv            # Per-session R² scores
```

### 2. `tta_random_folds_per_session.sh` - Per-Session TTA Mode ⭐

Runs TTA on each held-out session individually. Slower but provides comprehensive adapted model components for every session.

**Usage:**
```bash
./tta_random_folds_per_session.sh <path_to_random_folds_results>
```

**Example:**
```bash
./tta_random_folds_per_session.sh random_folds_20260108_123456
```

**Features:**
- Processes each held-out session individually (20 sessions × 20 folds = 400 TTA runs)
- Uses multi-GPU with round-robin scheduling for maximum parallelization
- Saves adapted model components for EVERY session separately
- No risk of results being overwritten (each session has its own directory)
- More comprehensive but slower execution

**Output Structure:**
```
random_folds_TIMESTAMP/
  tta_per_session_TIMESTAMP/
    fold0/
      MonkeyG_20150914_Session1_S1/
        tta_support_*.json
        adapted_models/          # Adapted model for this specific session
          fold0_support5000_coadapt/
            model_adapted.torch
            embeddings_stim_adapted.torch
            metadata.torch
          fold0_support5000_maml/
            ...
      MonkeyJ_20160426_Session1_S1/
        ...
    fold1/
      ...
    tta_timing_log.txt           # Detailed timing for all 400 runs
    tta_summary.csv              # Comprehensive per-session results
```

## Parameters

Both scripts use the following TTA parameters:

- **Support Size:** 5000 samples
- **TTA Epochs:** 7001
- **Strategies:** Co-adaptation and MAML
- **Progressive Unfreezing:** Enabled (threshold: 0)
- **Unfreeze Bases:** Enabled
- **GPUs:** 0 and 1 (multi-GPU mode)

## Session Batching Behavior

The TTA implementation automatically handles session batching differently based on progressive unfreezing:

### With Progressive Unfreezing Enabled (Default)
When `--unfreeze-bases` or `--unfreeze-basis-weights` is used with `--progressive-unfreezing-threshold 0`:

- **Batch Size:** 1 session at a time
- **Reason:** Progressive unfreezing adapts shared model components (basis generator), so each session needs independent processing
- **Output:** Each session gets its own subfolder with adapted model components
  ```
  adapted_models/
    model_support5000_coadapt/
      MonkeyG_20150914_Session1_S1/
        model_adapted.torch
        embeddings_stim_adapted.torch
        metadata.torch
      MonkeyJ_20160426_Session1_S1/
        model_adapted.torch
        embeddings_stim_adapted.torch
        metadata.torch
      ...
  ```
- **Result:** Every session has its own adapted model with session-specific shared components

### With Progressive Unfreezing Disabled
When progressive unfreezing is not enabled or threshold is not met:

- **Batch Size:** 5-15 sessions at a time (for memory efficiency)
- **Reason:** Only session-specific embeddings differ; shared components are unchanged
- **Output:** Single merged model for all sessions
  ```
  adapted_models/
    model_support5000_coadapt/
      model_adapted.torch           # Shared for all sessions
      embeddings_stim_adapted.torch # Contains embeddings for all sessions
      metadata.torch                # Lists all sessions processed
  ```
- **Behavior:**
  - First batch: Creates adapted model and embeddings
  - Subsequent batches: Load previous embeddings, merge new session embeddings, save updated file
  - Final result: Single model with embeddings for all sessions accumulated across batches

## Output Files

### Summary CSV (`tta_summary.csv`)
Contains per-session results with columns:
- `fold`: Fold number (0-19)
- `session_id`: Session identifier
- `strategy`: TTA strategy (coadapt or maml)
- `support_size`: Support set size (5000)
- `r2`: R² score for this session
- `start_time`: Job start timestamp
- `end_time`: Job end timestamp
- `duration_sec`: Duration in seconds
- `status`: SUCCESS, FAILED, etc.

### Timing Log (`tta_timing_log.txt`)
Detailed timing information for all TTA runs, including:
- Start/end times for each job
- Duration statistics
- Success/failure status

### Adapted Models
For each session (per-session mode) or fold (batch mode):
- `model_adapted.torch`: Adapted TBFM model
- `embeddings_stim_adapted.torch`: Adapted stimulus embeddings
- `metadata.torch`: Metadata including R², session IDs, hyperparameters

## Which Script to Use?

### Use `tta_random_folds.sh` (Batch Mode) if:
- You want faster execution
- You only need per-session R² scores for analysis
- You're running a quick validation experiment
- You're limited on compute time

### Use `tta_random_folds_per_session.sh` (Per-Session Mode) if: ⭐
- You need adapted model components for every individual session
- You're doing detailed per-session analysis
- You want to study how adaptation varies across sessions
- You need maximum flexibility for downstream analysis
- **This is recommended for comprehensive cross-validation**

## Example Workflow

### 1. Train Random Folds
```bash
# Train 20 folds with random session selections
./train_random_folds.sh
# Output: random_folds_20260108_123456/
```

### 2. Run TTA (Comprehensive Mode)
```bash
# Run TTA on all held-out sessions with per-session adapted models
./tta_random_folds_per_session.sh random_folds_20260108_123456
# Output: random_folds_20260108_123456/tta_per_session_20260108_140000/
```

### 3. Analyze Results
```python
import pandas as pd

# Load per-session results
results = pd.read_csv('random_folds_20260108_123456/tta_per_session_20260108_140000/tta_summary.csv')

# Compute mean R² per strategy
print(results.groupby('strategy')['r2'].describe())

# Analyze per-fold performance
print(results.groupby('fold')['r2'].mean())

# Find best/worst sessions
print(results.nlargest(10, 'r2'))
print(results.nsmallest(10, 'r2'))
```

## Performance Estimates

Based on typical execution times with progressive unfreezing enabled:

### Batch Mode (`tta_random_folds.sh`)
- ~400 TTA runs (20 sessions × 20 folds, batch size = 1 with progressive unfreezing)
- ~5-10 minutes per session
- **Total time: ~30-60 hours** (with 2 GPUs, parallel processing)
- Can be reduced with more GPUs

### Per-Session Mode (`tta_random_folds_per_session.sh`)
- ~400 TTA runs (20 sessions × 20 folds)
- ~5-10 minutes per session
- **Total time: ~30-60 hours** (with 2 GPUs, parallel processing)
- Can be reduced with more GPUs
- **Note:** Effectively same as batch mode when progressive unfreezing is enabled

### Without Progressive Unfreezing
If you disable progressive unfreezing (`--progressive-unfreezing-threshold 10000`):
- Batch size increases to 5-15 sessions
- **Batch mode:** ~20-40 batches, ~10-20 hours total
- **Per-session mode:** Still 400 runs, ~30-60 hours total

## Error Handling

Both scripts:
- Continue processing if individual jobs fail
- Log failures to the timing log and summary CSV
- Mark failed jobs with status codes
- Don't exit early (process all folds even if some fail)

## GPU Management

Both scripts:
- Use round-robin GPU assignment
- Automatically wait for GPUs to free up
- Clear GPU memory between jobs
- Support multi-GPU parallelization

## Tips

1. **Monitor Progress:** Check the timing log in real-time:
   ```bash
   tail -f random_folds_*/tta_*/tta_timing_log.txt
   ```

2. **Check GPU Usage:**
   ```bash
   watch nvidia-smi
   ```

3. **Resume Failed Jobs:** If a script fails partway through, you can manually run TTA on specific folds:
   ```bash
   python tta_testing.py \
       --model-paths fold5:random_folds_20260108_123456/fold5 \
       --adapt-session MonkeyG_20150914_Session1_S1 \
       --support-sizes 5000 \
       --tta-epochs 7001 \
       --output-dir random_folds_20260108_123456/tta_per_session_*/fold5/MonkeyG_20150914_Session1_S1
   ```

4. **Verify Completeness:** Check that all expected sessions were processed:
   ```python
   import pandas as pd
   df = pd.read_csv('tta_summary.csv')
   print(f"Expected: 400 (20 folds × 20 sessions)")
   print(f"Completed: {len(df[df['status'] == 'SUCCESS'])}")
   ```

## Troubleshooting

### Out of Memory Errors
- Reduce `--tta-epochs` to 5001
- Process fewer sessions in parallel (modify GPU assignment logic)
- Use only one GPU at a time

### Missing Sessions
- Verify all sessions have rest embeddings cached
- Check `hisi.torch` files are present in fold directories
- Review the timing log for error messages

### Slow Performance
- Use batch mode instead of per-session mode
- Increase number of GPUs with `--gpu-ids`
- Reduce TTA epochs for faster iteration

## Related Files

- `train_random_folds.sh`: First half of cross-validation (training)
- `tta_testing.py`: Core TTA evaluation script
- `train_session_count_sweep.sh`: Reference for multi-GPU training patterns
