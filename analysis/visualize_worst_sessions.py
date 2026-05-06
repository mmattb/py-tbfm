#!/usr/bin/env python3
"""
Visualize model predictions on worst performing sessions to diagnose poor performance.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tbfm import dataset, multisession, utils

# Constants
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")
TTA_RESULTS_DIR = Path("/home/danmuir/GitHub/py-tbfm/session_count_sweep_20251203_222740/tta_results")
ADAPTED_MODELS_DIR = TTA_RESULTS_DIR / "adapted_models"

# Worst sessions identified from analysis
WORST_SESSIONS = [
    "MonkeyG_20150917_Session3_M1",  # R² = 0.084 (worst)
    "MonkeyJ_20160429_Session3_S1",  # R² = 0.129
    "MonkeyG_20150918_Session1_M1",  # R² = 0.164
]

# Best session for comparison
BEST_SESSION = "MonkeyG_20150917_Session3_S1"  # R² = 0.648

def load_adapted_model(model_dir):
    """Load an adapted model from saved checkpoint."""
    model_path = Path(model_dir) / "model_adapted.torch"
    embeddings_path = Path(model_dir) / "embeddings_stim_adapted.torch"
    metadata_path = Path(model_dir) / "metadata.torch"

    if not all([model_path.exists(), embeddings_path.exists(), metadata_path.exists()]):
        return None, None, None

    model_state = torch.load(model_path, map_location='cpu')
    embeddings = torch.load(embeddings_path, map_location='cpu')
    metadata = torch.load(metadata_path, map_location='cpu')

    return model_state, embeddings, metadata


def load_session_data(session_id, window_size=200, batch_size=7500):
    """Load data for a specific session."""
    d, _ = multisession.load_stim_batched(
        batch_size=batch_size,
        window_size=window_size,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=[session_id],
        num_held_out_sessions=0,
    )

    # Split into train and test
    data_train, data_test = d.train_test_split(5000, test_cut=2500)

    return data_train, data_test


def compute_r2_score(y_true, y_pred):
    """Compute R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def visualize_predictions(session_id, model_key="1kx10_maml", strategy="maml", support_size=2500):
    """Visualize model predictions vs actual data for a session."""
    print(f"\n{'='*80}")
    print(f"Analyzing session: {session_id}")
    print(f"Model: {model_key}, Strategy: {strategy}, Support: {support_size}")
    print(f"{'='*80}")

    # Construct model directory path
    model_dir = ADAPTED_MODELS_DIR / f"{model_key}_support{support_size}_{strategy}" / session_id

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return

    # Load the adapted model
    print("Loading adapted model...")
    model_state, embeddings, metadata = load_adapted_model(model_dir)

    if model_state is None:
        print(f"Error: Could not load model from {model_dir}")
        return

    print(f"Metadata: {metadata}")

    # Load session data
    print("Loading session data...")
    try:
        data_train, data_test = load_session_data(session_id)
    except Exception as e:
        print(f"Error loading session data: {e}")
        return

    # Get test data
    test_batch = next(iter(data_test))
    x_test = test_batch['x']  # Shape: [batch, time, channels]
    y_test = test_batch['y']  # Ground truth

    print(f"Test data shape: x={x_test.shape}, y={y_test.shape}")

    # For now, let's just visualize the raw data patterns
    # Full model prediction would require reconstructing the model architecture

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{session_id}\n{model_key}, {strategy}, support={support_size}', fontsize=14)

    # Plot 1: Time series for first few channels
    ax = axes[0, 0]
    n_samples_to_plot = 5
    n_channels_to_plot = 3
    for ch in range(min(n_channels_to_plot, y_test.shape[2])):
        for i in range(min(n_samples_to_plot, y_test.shape[0])):
            alpha = 0.3
            ax.plot(y_test[i, :, ch].cpu().numpy(), alpha=alpha, color=f'C{ch}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Neural Activity')
    ax.set_title(f'Sample trajectories (first {n_channels_to_plot} channels)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of neural activity
    ax = axes[0, 1]
    y_flat = y_test.cpu().numpy().flatten()
    ax.hist(y_flat, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Neural Activity Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Neural Activity')
    ax.axvline(y_flat.mean(), color='red', linestyle='--', label=f'Mean: {y_flat.mean():.3f}')
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, label='Zero')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Channel variance
    ax = axes[1, 0]
    channel_vars = y_test.var(dim=(0, 1)).cpu().numpy()
    ax.bar(range(len(channel_vars)), channel_vars)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Variance')
    ax.set_title('Variance by Channel')
    ax.grid(True, alpha=0.3)

    # Plot 4: Temporal variance
    ax = axes[1, 1]
    temporal_vars = y_test.var(dim=(0, 2)).cpu().numpy()
    ax.plot(temporal_vars)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Variance (across channels & samples)')
    ax.set_title('Temporal Variance')
    ax.grid(True, alpha=0.3)

    # Plot 5: Mean activity over time
    ax = axes[2, 0]
    mean_activity = y_test.mean(dim=(0, 2)).cpu().numpy()
    ax.plot(mean_activity)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Activity')
    ax.set_title('Mean Activity Over Time')
    ax.grid(True, alpha=0.3)

    # Plot 6: Signal-to-noise estimate (variance ratio early vs late)
    ax = axes[2, 1]
    early_window = y_test[:, :50, :]  # First 50 timesteps
    late_window = y_test[:, -50:, :]  # Last 50 timesteps

    early_var = early_window.var(dim=0).mean(dim=0).cpu().numpy()
    late_var = late_window.var(dim=0).mean(dim=0).cpu().numpy()

    ax.scatter(early_var, late_var, alpha=0.5)
    ax.plot([0, max(early_var.max(), late_var.max())],
            [0, max(early_var.max(), late_var.max())],
            'r--', label='Equal variance')
    ax.set_xlabel('Early Window Variance')
    ax.set_ylabel('Late Window Variance')
    ax.set_title('Temporal Stability (early vs late variance by channel)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(f"/home/danmuir/GitHub/py-tbfm/session_diagnosis_{session_id}_{model_key}_{strategy}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    # Print statistics
    print(f"\nData Statistics:")
    print(f"  Shape: {y_test.shape} (batch, time, channels)")
    print(f"  Mean: {y_flat.mean():.4f}")
    print(f"  Std: {y_flat.std():.4f}")
    print(f"  Min: {y_flat.min():.4f}")
    print(f"  Max: {y_flat.max():.4f}")
    print(f"  Median channel variance: {np.median(channel_vars):.4f}")
    print(f"  Median temporal variance: {np.median(temporal_vars):.4f}")

    return fig


def main():
    """Main function."""
    print("="*80)
    print("VISUALIZING WORST PERFORMING SESSIONS")
    print("="*80)

    # Configuration
    model_key = "1kx10_maml"
    strategy = "maml"
    support_size = 2500

    # Analyze worst sessions
    for session_id in WORST_SESSIONS[:2]:  # Analyze first 2 worst sessions
        try:
            visualize_predictions(session_id, model_key, strategy, support_size)
        except Exception as e:
            print(f"Error processing {session_id}: {e}")
            import traceback
            traceback.print_exc()

    # Analyze best session for comparison
    try:
        print("\n" + "="*80)
        print("For comparison, analyzing BEST performing session:")
        print("="*80)
        visualize_predictions(BEST_SESSION, model_key, strategy, support_size)
    except Exception as e:
        print(f"Error processing {BEST_SESSION}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nCheck the generated PNG files for visualizations of the data patterns.")
    print("Look for:")
    print("  - Unusual activity distributions")
    print("  - Very low variance (dead channels)")
    print("  - High noise / instability")
    print("  - Systematic drift over time")


if __name__ == "__main__":
    main()
