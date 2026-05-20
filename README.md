# py-tbfm
PyTorch implementation of the Temporal Basis Function Model (TBFM) for neural time-series forecasting under stimulation.

## Installation

```
pip install .
```

## Single-session quick start

Instantiate and train a TBFM on a single recording session:

```python
from tbfm import tbfm as tbfm_module

FORECAST_HORIZON = TRIAL_LENGTH - RUNWAY_LENGTH

model = tbfm_module.TBFM(
    NUM_CHANNELS,              # Dimensionality of time series
    STIM_DESC_DIM,             # Dimensionality of stimulation descriptor
    RUNWAY_LENGTH,             # Length of runway, in time steps
    NUM_BASES,                 # Number of temporal bases to learn
    FORECAST_HORIZON,          # Length of forecast, in time steps
    batchy=y_train,            # Training dataset, used to estimate means/stdevs
    latent_dim=LATENT_DIM,     # Latent dimension of the basis generator network
    basis_depth=BASIS_DEPTH,   # Depth of the basis generator network
    device=DEVICE,             # e.g. "cpu" or "cuda:0"
)
optim = model.get_optim(lr=2e-4)
```

A forward pass:

```python
yhat = model(
    runways,           # (batch_size, RUNWAY_LENGTH, NUM_CHANNELS)
    stim_descriptor,   # (batch_size, FORECAST_HORIZON, STIM_DESC_DIM)
)
# yhat: (batch_size, FORECAST_HORIZON, NUM_CHANNELS)
```

## Multi-session training

For multi-session meta-learning, use the `tma_standalone.py` training script. It trains a shared TBFM across many sessions using MAML-style inner-loop adaptation, with per-session learnable stimulus embeddings.

**Required environment variable:**
```bash
export TBFM_DATA_DIR=/path/to/your/data
```

The data directory should contain one subdirectory per session (e.g. `MonkeyG_20150914_Session1_S1/`), each with a `torchraw/` subdirectory containing pre-processed trial tensors.

**Basic usage:**
```bash
python tma_standalone.py NUM_BASES NUM_SESSIONS GPU_ID COADAPT BASIS_RESIDUAL_RANK TRAIN_SIZE SHUFFLE
```

Key arguments:
- `NUM_BASES` — number of temporal bases (e.g. `100`)
- `NUM_SESSIONS` — number of sessions to include in training
- `GPU_ID` — GPU index, or `-1` for CPU
- `COADAPT` — `true` to use co-adaptation (trains per-session embeddings via outer loop); `false` for MAML inner-loop adaptation
- `BASIS_RESIDUAL_RANK` — rank of the per-session basis residual (LoRA-style correction); `0` disables residual mode and uses concatenation instead
- `TRAIN_SIZE` — number of training trials per session
- `SHUFFLE` — `true` to randomly sample support sets each epoch (recommended)

See `python tma_standalone.py --help` for the full list of options, including ablation flags.

## Demos and walkthroughs

- **`TBFM Demo.ipynb`** — single-session walkthrough using synthetic data
- **`TBFM FSAM Demo.ipynb`** — builds the TBFM via forward stagewise additive modeling; recommended after the first demo
- **`TBFM Traveling Wave Demo.ipynb`** — demonstrates TBFM applied to traveling wave data
- **`TBFM Multisession Demo.ipynb`** — full multi-session workflow: pretraining across synthetic sessions and test-time adaptation to held-out sessions

## Architecture

![detail_arch](https://github.com/user-attachments/assets/daf3fb08-f087-4dcb-b4fb-5835a2f8f5c0)

