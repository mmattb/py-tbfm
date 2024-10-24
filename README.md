# py-tbfm
Implementation of the Temporal Basis Function Model (TBFM)

## Quick start
If you are looking to use the TBFM implementation, simply install this module; e.g.:

```
pip install .
```

then use it as follows:

```
model = tbfm.TBFM(NUM_CHANNELS, STIM_DESC_DIM, RUNWAY_LENGTH, NUM_BASES, TRIAL_LENGTH-RUNWAY_LENGTH,
                  batchy=y_train,
                  latent_dim=LATENT_DIM,
                  basis_depth=BASIS_DEPTH,
                  device=DEVICE)
optim = model.get_optim(lr=2e-4)
```

## Walkthrough and demo

``TBFM Demo.ipynb`` provides a walkthrough which uses some synthetic data.
