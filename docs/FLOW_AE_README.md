# Normalizing Flow Autoencoder (FlowChannelAE)

An **invertible** MLP-based autoencoder using normalizing flows (RealNVP-style affine coupling layers).

## Key Features

### 1. **Exact Invertibility**
Unlike standard autoencoders, this model is **perfectly invertible**:
- `decode(encode(x)) == x` (up to numerical precision ~1e-7)
- Uses affine coupling layers from RealNVP
- No information loss in the latent representation

### 2. **Variable Channel Support**
Like `LinearChannelAE`, supports different channel counts per session:
```python
# Session 1: 60 channels
z1 = model.encode(x1, mask=torch.arange(60))

# Session 2: 80 channels
z2 = model.encode(x2, mask=torch.arange(80))
```

### 3. **Nonlinear Transformations**
Uses MLP networks internally for more expressive representations than linear PCA.

## Architecture

```
Input (in_dim)
    ↓
Linear Projection (optional) → latent_dim
    ↓
Coupling Layer 1 + Permutation
    ↓
Coupling Layer 2 + Permutation
    ↓
...
    ↓
Latent representation (latent_dim)
```

Each coupling layer applies:
```
x1, x2 = split(x)
z1 = x1
z2 = x2 * exp(scale_net(x1)) + translate_net(x1)
```

This is **invertible** because we can solve for `x2`:
```
x2 = (z2 - translate_net(x1)) * exp(-scale_net(x1))
```

## Usage

### Basic Usage

```python
from tbfm.flow_ae import FlowChannelAE

# Create model
model = FlowChannelAE(
    in_dim=100,           # Max channels across sessions
    latent_dim=50,        # Compressed dimension
    num_flow_layers=4,    # More = more expressive
    hidden_dim=128,       # MLP hidden layer size
    use_linear_projection=True,
    device='cuda'
)

# Encode
mask = torch.arange(num_channels)  # Which channels are present
z = model.encode(x, mask)

# Decode (exact inverse!)
x_recon = model.decode(z, mask)

# Should be nearly zero
error = (x - x_recon).abs().max()  # ~1e-7
```

### Using with Hydra Config

Update your config to use the flow autoencoder:

```yaml
# In your config file or override
defaults:
  - ae: flow  # Instead of ae: default

ae:
  module:
    latent_dim: 50
    num_flow_layers: 6     # More layers = more expressive
    hidden_dim: 128        # Larger = more capacity
```

Or override from command line:
```bash
python train.py ae=flow ae.module.num_flow_layers=6
```

### In Your Training Notebook

```python
# In train_multisession_all.ipynb, just change the config:
with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
    cfg = compose(
        config_name="config",
        overrides=["ae=flow"]  # Use flow autoencoder
    )

# Everything else stays the same!
ms = multisession.build_from_cfg(cfg, data_train, device=DEVICE)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_dim` | required | Maximum input dimension |
| `latent_dim` | required | Latent space dimension |
| `num_flow_layers` | 4 | Number of coupling layers (more = better but slower) |
| `hidden_dim` | 128 | Hidden layer width in coupling networks |
| `use_linear_projection` | True | Whether to use linear projection layer |
| `use_lora` | False | LoRA adaptation (for API compatibility; not yet implemented) |

**Recommendations:**
- Start with `num_flow_layers=4`, increase to 6-8 for more capacity
- Use `hidden_dim=128` for small datasets, 256+ for large
- Set `use_linear_projection=True` if `latent_dim < in_dim`

## Testing

Run the test suite:
```bash
cd examples
python test_flow_ae.py
```

This verifies:
- ✓ Invertibility (error < 1e-5)
- ✓ Variable channel masking
- ✓ Gradient flow
- ✓ Log determinant computation

## Comparison with Linear AE

| Feature | LinearChannelAE | FlowChannelAE |
|---------|----------------|---------------|
| **Invertibility** | No (lossy) | Yes (exact) |
| **Expressiveness** | Linear | Nonlinear |
| **Parameters** | O(in_dim × latent_dim) | O(num_layers × latent_dim × hidden_dim) |
| **Speed** | Faster | Slower (more layers) |
| **Interpretability** | High (like PCA) | Lower |

**When to use Flow AE:**
- Need exact invertibility for analysis
- Data has nonlinear structure
- Want more expressive latent representations
- Can afford extra computation

**When to use Linear AE:**
- Want PCA-like interpretability
- Need maximum speed
- Linear structure is sufficient
- Smaller model size preferred

## Advanced: Log Determinant

For density estimation or variational inference:

```python
z, log_det = model.encode(x, mask, return_log_det=True)

# log p(x) = log p(z) + log_det
log_prior = -0.5 * (z**2).sum(-1)  # Standard normal
log_px = log_prior + log_det
```

## Troubleshooting

**Reconstruction error too high:**
- Increase `num_flow_layers` (try 6-8)
- Increase `hidden_dim` (try 256)
- Check numerical precision (use float32)

**Training unstable:**
- Reduce learning rate (try 1e-4 or 1e-5)
- Add gradient clipping
- Reduce `num_flow_layers`

**Too slow:**
- Reduce `num_flow_layers` to 2-3
- Reduce `hidden_dim` to 64
- Use smaller batch size

## Implementation Details

- **Initialization:** Coupling layers initialized near identity mapping
- **Permutations:** Random permutations between layers to mix all dimensions
- **Stability:** Tanh on scale network keeps values bounded
- **Warm start:** Supports PCA initialization for linear projection layer

## References

- RealNVP: [Dinh et al. 2016](https://arxiv.org/abs/1605.08803)
- Normalizing Flows: [Papamakarios et al. 2019](https://arxiv.org/abs/1912.02762)
