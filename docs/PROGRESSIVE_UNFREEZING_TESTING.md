# Progressive Unfreezing Test Suite

This directory contains scripts to test progressive unfreezing (supervised fine-tuning) at high support sizes in TTA.

## Hypothesis

At high support sizes (2000+), supervised fine-tuning of basis weights and/or the basis generator should improve performance because:
1. More data is available to learn session-specific adaptations
2. The meta-learned initialization provides a good starting point
3. Low learning rates prevent catastrophic forgetting

## Test Scripts

### 1. Quick Test (Recommended First)
```bash
./test_progressive_unfreezing_quick.sh
```

**Purpose**: Fast validation that the feature works correctly
- Uses only 1000 epochs (vs 7001 for full test)
- Tests only support size 2500
- Compares baseline vs full unfreezing
- **Runtime**: ~10-20 minutes

### 2. Full Test Suite
```bash
./test_progressive_unfreezing.sh
```

**Purpose**: Comprehensive evaluation across support sizes and strategies
- Uses full 7001 epochs
- Tests support sizes: 1000, 2500, 5000
- Tests 4 strategies:
  1. Baseline (no unfreezing)
  2. Unfreeze basis weights only
  3. Unfreeze bases only
  4. Unfreeze both
- **Runtime**: ~2-3 hours

### 3. Analyze Results
```bash
python analyze_progressive_unfreezing.py data/tta_progressive_unfreezing
```

**Purpose**: Compare R² scores across strategies
- Prints summary tables
- Shows improvement over baseline
- Identifies best strategy per support size
- Saves CSV for further analysis

## Configuration

Edit the test scripts to customize:

```bash
# Which GPU to use
CUDA_DEVICE="1"

# Which session to adapt
ADAPT_SESSION="MonkeyG_20150914_Session3_S1"

# Which model to test
MODELS="100_25_inner_ts5000_shuffle"

# Support sizes to test
SUPPORT_SIZES="1000 2500 5000"

# Number of training epochs
TTA_EPOCHS=7001

# Progressive unfreezing threshold
--progressive-unfreezing-threshold 2000

# Learning rates
--basis-weight-lr 1e-5  # For basis weights
--bases-lr 1e-6         # For basis generator
```

## Command-Line Arguments

The `tta_testing.py` script supports these progressive unfreezing arguments:

- `--progressive-unfreezing-threshold N`: Support size threshold (default: 2000)
- `--unfreeze-basis-weights`: Enable fine-tuning of basis weights
- `--unfreeze-bases`: Enable fine-tuning of basis generator
- `--basis-weight-lr LR`: Learning rate for basis weights (default: 1e-5)
- `--bases-lr LR`: Learning rate for basis generator (default: 1e-6)

## Example Usage

### Manual test with custom parameters:
```bash
python tta_testing.py \
    --cuda-device 1 \
    --support-sizes 2500 5000 \
    --adapt-session MonkeyG_20150914_Session3_S1 \
    --tta-epochs 7001 \
    --models 100_25_inner_ts5000_shuffle \
    --output-dir data/custom_test \
    --progressive-unfreezing-threshold 2000 \
    --unfreeze-basis-weights \
    --unfreeze-bases \
    --basis-weight-lr 1e-5 \
    --bases-lr 1e-6 \
    --no-plot-display
```

### Compare results:
```bash
python plot_tta_results.py \
    data/tta_progressive_unfreezing/baseline/tta_*.json \
    data/tta_progressive_unfreezing/unfreeze_both/tta_*.json
```

## Expected Results

If progressive unfreezing helps, you should see:
- Higher R² at support sizes ≥ 2000 with unfreezing enabled
- Minimal difference at support sizes < 2000 (threshold not reached)
- Best results likely from unfreezing basis weights (more direct impact)

## Troubleshooting

### No improvement observed
- Try different learning rates (e.g., 1e-4, 1e-6)
- Adjust threshold (try 1000 or 3000)
- Check if training is converging (look at loss curves)

### Out of memory errors
- Reduce support sizes
- Use smaller batch sizes
- Enable gradient checkpointing

### Slow execution
- Run quick test first to validate
- Use multi-GPU mode: `--use-multi-gpu`
- Reduce number of epochs for initial testing

## Files Created

After running tests, you'll have:
```
data/tta_progressive_unfreezing/
├── baseline/
│   ├── tta_TIMESTAMP.json
│   ├── tta_TIMESTAMP.log
│   └── tta_TIMESTAMP.png
├── unfreeze_basis_weights/
│   └── ...
├── unfreeze_bases/
│   └── ...
├── unfreeze_both/
│   └── ...
└── progressive_unfreezing_comparison.csv
```

## Next Steps

1. **Run quick test**: `./test_progressive_unfreezing_quick.sh`
2. **Analyze results**: `python analyze_progressive_unfreezing.py data/tta_progressive_unfreezing_quick`
3. **If promising, run full test**: `./test_progressive_unfreezing.sh`
4. **Compare with plots**: Use `plot_tta_results.py` to visualize
5. **Iterate on hyperparameters** if needed
