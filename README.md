# py-tbfm
Implementation of the Temporal Basis Function Model (TBFM)

## Quick start
If you are looking to use the TBFM implementation, simply install this module; e.g.:

```
pip install .
```

then use it as follows:

```
model = tbfm.TBFM(NUM_CHANNELS,              # Dimensionality of time series
                  STIM_DESC_DIM,             # Dimensionality of stimulation descriptor
                  RUNWAY_LENGTH,             # Length of runway, in time steps
                  NUM_BASES,                 # Number of bases we will learn
                  FORECAST_HORIZON,          # Length of forecast, in time steps
                  batchy=y_train,            # A training dataset, for estimating means/stdevs
                  latent_dim=LATENT_DIM,     # Latent dimension of basis generator network
                  basis_depth=BASIS_DEPTH,   # Depth of basis generator network
                  device=DEVICE)             # Choice of device, e.g. "cpu" or "cuda:0"
optim = model.get_optim(lr=2e-4)             # Optimizer for use in training
```

A forward pass looks like this:
```
FORECAST_HORIZON = TRIAL_LENGTH - RUNWAY_LENGTH

yhat = model(
             runways,            # tensor shaped (batch_size, RUNWAY_LENGTH, NUM_CHANNELS)
             stim_descriptor,    # tensor shaped (batch_size, FORECAST_HORIZON, STIM_DESC_DIM)
       )

# yhat is a tensor shaped (batch_size, FORECAST_HORIZON, NUM_CHANNELS)
```

## Walkthrough and demo

``TBFM Demo.ipynb`` provides a detailed walkthrough which uses some synthetic data.
``TBFM FSAM Demo.ipynb`` provides an additional demo where we build the TBFM using forward stagewise additive modeling. It's recommended to go through this one after the first.

## Architecture

![detail_arch](https://github.com/user-attachments/assets/daf3fb08-f087-4dcb-b4fb-5835a2f8f5c0)

