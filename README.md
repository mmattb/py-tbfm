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

## Walkthrough and demo

``TBFM Demo.ipynb`` provides a walkthrough which uses some synthetic data.
