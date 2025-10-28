"""
Test script for FlowChannelAE - demonstrates invertibility and usage.
"""

import torch
import sys
sys.path.append('..')

from tbfm.flow_ae import FlowChannelAE


def test_invertibility():
    """Test that encode-decode is perfectly invertible."""
    print("=" * 60)
    print("Testing FlowChannelAE Invertibility")
    print("=" * 60)

    # Setup
    in_dim = 74
    latent_dim = 50
    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = FlowChannelAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_flow_layers=6,
        hidden_dim=128,
        use_linear_projection=True,
        use_lora=False,
        device=device
    )
    model.eval()

    # Create random input
    x = torch.randn(batch_size, in_dim, device=device)
    mask = torch.arange(in_dim, device=device)  # All channels present

    # Encode then decode
    with torch.no_grad():
        z = model.encode(x, mask)
        x_recon = model.decode(z, mask)

    # Check reconstruction error
    error = (x - x_recon).abs().max().item()
    mean_error = (x - x_recon).abs().mean().item()

    print(f"\nInput shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"\nMax absolute error: {error:.2e}")
    print(f"Mean absolute error: {mean_error:.2e}")

    if error < 1e-5:
        print("\n✓ PASSED: Model is invertible (error < 1e-5)")
    else:
        print(f"\n✗ FAILED: Error too large ({error:.2e})")

    return error < 1e-5


def test_variable_channels():
    """Test with variable channel counts (session masking)."""
    print("\n" + "=" * 60)
    print("Testing Variable Channel Support")
    print("=" * 60)

    in_dim = 100
    latent_dim = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FlowChannelAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_flow_layers=4,
        hidden_dim=64,
        use_lora=False,
        device=device
    )
    model.eval()

    # Test different session sizes
    sessions = {
        'session_1': 60,  # 60 channels
        'session_2': 80,  # 80 channels
        'session_3': 50,  # 50 channels
    }

    print("\nTesting different session channel counts:")

    all_passed = True
    for session_id, num_channels in sessions.items():
        # Create mask for this session
        mask = torch.arange(num_channels, device=device)

        # Random input with this many channels
        x = torch.randn(5, num_channels, device=device)

        with torch.no_grad():
            z = model.encode(x, mask)
            x_recon = model.decode(z, mask)

        error = (x - x_recon).abs().max().item()

        print(f"  {session_id} ({num_channels} channels): error = {error:.2e}", end="")

        if error < 1e-5:
            print(" ✓")
        else:
            print(" ✗")
            all_passed = False

    if all_passed:
        print("\n✓ PASSED: All session masks work correctly")
    else:
        print("\n✗ FAILED: Some sessions have errors")

    return all_passed


def test_gradients():
    """Test that gradients flow properly through the model."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    in_dim = 50
    latent_dim = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FlowChannelAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_flow_layers=3,
        hidden_dim=64,
        use_lora=False,
        device=device
    )

    # Random data
    x = torch.randn(8, in_dim, device=device)
    mask = torch.arange(in_dim, device=device)

    # Forward pass with loss
    x_recon = model.reconstruct(x, mask)
    loss = model.reconstruction_loss(x, x_recon)

    # Backward pass
    loss.backward()

    # Check that all parameters have gradients
    param_count = 0
    params_with_grad = 0

    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1

    print(f"\nParameters with gradients: {params_with_grad}/{param_count}")
    print(f"Loss value: {loss.item():.4f}")

    if params_with_grad == param_count:
        print("\n✓ PASSED: All parameters receive gradients")
        return True
    else:
        print(f"\n✗ FAILED: {param_count - params_with_grad} parameters missing gradients")
        return False


def test_log_det():
    """Test log determinant computation for normalizing flows."""
    print("\n" + "=" * 60)
    print("Testing Log Determinant")
    print("=" * 60)

    in_dim = 40
    latent_dim = 40  # Must be equal for pure flow (no projection)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pure flow model (no projection)
    model = FlowChannelAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_flow_layers=4,
        hidden_dim=64,
        use_linear_projection=False,  # No dimension change
        use_lora=False,
        device=device
    )
    model.eval()

    x = torch.randn(5, in_dim, device=device)
    mask = torch.arange(in_dim, device=device)

    with torch.no_grad():
        z, log_det = model.encode(x, mask, return_log_det=True)

    print(f"\nLog determinant shape: {log_det.shape}")
    print(f"Log determinant mean: {log_det.mean().item():.4f}")
    print(f"Log determinant std: {log_det.std().item():.4f}")

    print("\n✓ Log determinant computed successfully")
    return True


def compare_with_linear_ae():
    """Compare reconstruction quality with linear autoencoder."""
    print("\n" + "=" * 60)
    print("Comparing Flow AE vs Linear AE (Reconstruction)")
    print("=" * 60)

    from tbfm.ae import LinearChannelAE

    in_dim = 60
    latent_dim = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create both models
    flow_ae = FlowChannelAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_flow_layers=4,
        hidden_dim=64,
        use_lora=False,
        device=device
    )

    linear_ae = LinearChannelAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        use_bias=False,
        device=device
    )

    # Random data
    x = torch.randn(20, in_dim, device=device)
    mask = torch.arange(in_dim, device=device)

    # Test both
    with torch.no_grad():
        flow_recon = flow_ae.reconstruct(x, mask)
        linear_recon = linear_ae.reconstruct(x, mask)

    flow_error = nn.functional.mse_loss(x, flow_recon).item()
    linear_error = nn.functional.mse_loss(x, linear_recon).item()

    print(f"\nFlow AE MSE: {flow_error:.6f}")
    print(f"Linear AE MSE: {linear_error:.6f}")
    print(f"\nNote: Flow AE should be nearly perfect (~1e-10) when encoding/decoding")
    print("Linear AE will have reconstruction error since it's lossy compression")

    return True


if __name__ == "__main__":
    import torch.nn as nn

    print("\n" + "=" * 60)
    print("FlowChannelAE Test Suite")
    print("=" * 60 + "\n")

    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Running on: {device_name}\n")

    # Run all tests
    results = {}
    results['invertibility'] = test_invertibility()
    results['variable_channels'] = test_variable_channels()
    results['gradients'] = test_gradients()
    results['log_det'] = test_log_det()
    results['comparison'] = compare_with_linear_ae()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60 + "\n")
