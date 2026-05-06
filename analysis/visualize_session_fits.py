#!/usr/bin/env python3
"""
Visualize model fit quality on individual sessions from adapted models.
"""

import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from tbfm import multisession

# Constants
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")
MODEL_DIR = Path("/home/danmuir/GitHub/py-tbfm/session_count_sweep_20251203_222740/tta_results/adapted_models/1kx25_maml_support5000_maml")
BASE_MODEL_DIR = Path("/home/danmuir/GitHub/py-tbfm/session_count_sweep_20251203_222740/1kx25_maml")


def compute_r2_per_channel(y_true, y_pred):
    """Compute R² score per channel."""
    # y_true, y_pred shape: [batch, time, channels]
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=(0, 1))  # Sum over batch and time
    ss_tot = torch.sum((y_true - y_true.mean(dim=(0, 1), keepdim=True)) ** 2, dim=(0, 1))
    r2 = 1 - ss_res / ss_tot
    return r2


def compute_r2_overall(y_true, y_pred):
    """Compute overall R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def visualize_session_fit(session_id, ms_model, embeddings, device='cpu'):
    """Visualize model fit for a specific session."""
    print(f"\n{'='*80}")
    print(f"Analyzing session: {session_id}")
    print(f"{'='*80}")

    # Load session data
    print("Loading session data...")
    d, _ = multisession.load_stim_batched(
        batch_size=7500,
        window_size=185,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=[session_id],
        num_held_out_sessions=0,
    )
    _, data_test = d.train_test_split(5000, test_cut=2500)

    # Get test batch
    test_batch = next(iter(data_test))
    x = test_batch['x'].to(device)
    y_true = test_batch['y'].to(device)
    session_embedding = embeddings[session_id].to(device)

    print(f"Data shapes: x={x.shape}, y={y_true.shape}, embedding={session_embedding.shape}")

    # Get predictions
    print("Computing predictions...")
    ms_model.eval()
    with torch.no_grad():
        # Create session embedding batch
        batch_size = x.shape[0]
        session_emb_batch = session_embedding.unsqueeze(0).expand(batch_size, -1)

        # Get predictions
        y_pred = ms_model(x, session_emb_batch)

    # Compute R² scores
    r2_overall = compute_r2_overall(y_true, y_pred)
    r2_per_channel = compute_r2_per_channel(y_true, y_pred).cpu().numpy()

    print(f"\nR² Scores:")
    print(f"  Overall: {r2_overall:.4f}")
    print(f"  Per channel - Mean: {r2_per_channel.mean():.4f}, Median: {np.median(r2_per_channel):.4f}")
    print(f"  Per channel - Min: {r2_per_channel.min():.4f}, Max: {r2_per_channel.max():.4f}")

    # Move to CPU for plotting
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'{session_id} - Overall R² = {r2_overall:.4f}', fontsize=16, fontweight='bold')

    # Plot 1: Sample trajectories for a few channels
    ax1 = fig.add_subplot(gs[0, :2])
    n_samples = 3
    n_channels_to_show = 3
    for ch in range(n_channels_to_show):
        for i in range(n_samples):
            ax1.plot(y_true_np[i, :, ch], alpha=0.6, color=f'C{ch}', linestyle='-', label='True' if i == 0 and ch == 0 else '')
            ax1.plot(y_pred_np[i, :, ch], alpha=0.6, color=f'C{ch}', linestyle='--', label='Pred' if i == 0 and ch == 0 else '')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neural Activity')
    ax1.set_title(f'Sample Trajectories (first {n_channels_to_show} channels, {n_samples} samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: R² per channel distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(r2_per_channel, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(r2_per_channel.mean(), color='red', linestyle='--', label=f'Mean: {r2_per_channel.mean():.3f}')
    ax2.axvline(np.median(r2_per_channel), color='green', linestyle='--', label=f'Median: {np.median(r2_per_channel):.3f}')
    ax2.set_xlabel('R² Score')
    ax2.set_ylabel('Number of Channels')
    ax2.set_title('R² Distribution Across Channels')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction vs True scatter (all timepoints, random subset of channels)
    ax3 = fig.add_subplot(gs[1, 0])
    n_channels_scatter = min(10, y_true_np.shape[2])
    sample_channels = np.random.choice(y_true_np.shape[2], n_channels_scatter, replace=False)
    y_true_flat = y_true_np[:, :, sample_channels].flatten()
    y_pred_flat = y_pred_np[:, :, sample_channels].flatten()

    # Subsample for plotting (too many points otherwise)
    n_points = min(10000, len(y_true_flat))
    indices = np.random.choice(len(y_true_flat), n_points, replace=False)

    ax3.scatter(y_true_flat[indices], y_pred_flat[indices], alpha=0.1, s=1)
    lims = [min(y_true_flat.min(), y_pred_flat.min()), max(y_true_flat.max(), y_pred_flat.max())]
    ax3.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax3.set_xlabel('True Value')
    ax3.set_ylabel('Predicted Value')
    ax3.set_title(f'Pred vs True ({n_channels_scatter} random channels)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Plot 4: Residuals over time
    ax4 = fig.add_subplot(gs[1, 1])
    residuals = y_true_np - y_pred_np
    residuals_mean_over_channels = residuals.mean(axis=2)  # Average over channels
    residuals_mean_over_time = residuals_mean_over_channels.mean(axis=0)  # Average over batch
    residuals_std_over_time = residuals_mean_over_channels.std(axis=0)

    time_steps = np.arange(len(residuals_mean_over_time))
    ax4.plot(time_steps, residuals_mean_over_time, 'b-', label='Mean residual')
    ax4.fill_between(time_steps,
                      residuals_mean_over_time - residuals_std_over_time,
                      residuals_mean_over_time + residuals_std_over_time,
                      alpha=0.3, label='±1 std')
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Residual (True - Pred)')
    ax4.set_title('Residuals Over Time (averaged across batch & channels)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Residual distribution
    ax5 = fig.add_subplot(gs[1, 2])
    residuals_flat = residuals.flatten()
    ax5.hist(residuals_flat, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax5.axvline(residuals_flat.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {residuals_flat.mean():.3f}')
    ax5.set_xlabel('Residual')
    ax5.set_ylabel('Count')
    ax5.set_title('Residual Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: R² by channel
    ax6 = fig.add_subplot(gs[2, :])
    ax6.bar(range(len(r2_per_channel)), r2_per_channel, alpha=0.7)
    ax6.axhline(r2_per_channel.mean(), color='red', linestyle='--', label=f'Mean: {r2_per_channel.mean():.3f}')
    ax6.set_xlabel('Channel Index')
    ax6.set_ylabel('R² Score')
    ax6.set_title('R² Score by Channel')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Predicted vs True for worst channel
    worst_channel = np.argmin(r2_per_channel)
    ax7 = fig.add_subplot(gs[3, 0])
    for i in range(min(5, y_true_np.shape[0])):
        ax7.plot(y_true_np[i, :, worst_channel], alpha=0.6, color='blue', label='True' if i == 0 else '')
        ax7.plot(y_pred_np[i, :, worst_channel], alpha=0.6, color='red', linestyle='--', label='Pred' if i == 0 else '')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Activity')
    ax7.set_title(f'Worst Channel (ch={worst_channel}, R²={r2_per_channel[worst_channel]:.3f})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Predicted vs True for best channel
    best_channel = np.argmax(r2_per_channel)
    ax8 = fig.add_subplot(gs[3, 1])
    for i in range(min(5, y_true_np.shape[0])):
        ax8.plot(y_true_np[i, :, best_channel], alpha=0.6, color='blue', label='True' if i == 0 else '')
        ax8.plot(y_pred_np[i, :, best_channel], alpha=0.6, color='red', linestyle='--', label='Pred' if i == 0 else '')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Activity')
    ax8.set_title(f'Best Channel (ch={best_channel}, R²={r2_per_channel[best_channel]:.3f})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Prediction quality over time
    ax9 = fig.add_subplot(gs[3, 2])
    r2_per_time = []
    for t in range(y_true_np.shape[1]):
        y_t = y_true_np[:, t, :]
        y_pred_t = y_pred_np[:, t, :]
        ss_res = np.sum((y_t - y_pred_t) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        r2_t = 1 - ss_res / ss_tot
        r2_per_time.append(r2_t)

    ax9.plot(r2_per_time)
    ax9.axhline(r2_overall, color='red', linestyle='--', label=f'Overall: {r2_overall:.3f}')
    ax9.set_xlabel('Time Step')
    ax9.set_ylabel('R² Score')
    ax9.set_title('R² Over Time (all channels)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    safe_session_name = session_id.replace('/', '_')
    output_path = Path(f"/home/danmuir/GitHub/py-tbfm/session_fit_{safe_session_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    return r2_overall, r2_per_channel


def main():
    """Main function."""
    print("="*80)
    print("VISUALIZING INDIVIDUAL SESSION FITS")
    print("="*80)

    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load metadata
    metadata = torch.load(MODEL_DIR / 'metadata.torch')
    print(f"\nModel metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    # Load adapted model and embeddings
    print("\nLoading adapted model...")
    model_state = torch.load(MODEL_DIR / 'model_adapted.torch', map_location=device)
    embeddings = torch.load(MODEL_DIR / 'embeddings_stim_adapted.torch', map_location=device)

    # Load configuration
    print("Loading configuration...")
    config_dir = Path("/home/danmuir/GitHub/py-tbfm/conf")

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return

    with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        cfg = compose(config_name="config")

    # Load and apply hyperparameters
    print("Loading hyperparameters...")
    hyperparams = torch.load(BASE_MODEL_DIR / "hyperparameters.torch")
    print(f"Hyperparameters: {hyperparams}")

    # Apply hyperparameters to config
    OmegaConf.set_struct(cfg, False)
    if 'latent_dim' in hyperparams:
        cfg.latent_dim = hyperparams['latent_dim']
        cfg.ae.module.latent_dim = hyperparams['latent_dim']
        cfg.tbfm.module.in_dim = hyperparams['latent_dim']
    if 'num_bases' in hyperparams:
        cfg.tbfm.module.num_bases = hyperparams['num_bases']
    if 'basis_residual_rank' in hyperparams:
        cfg.meta.basis_residual_rank = hyperparams['basis_residual_rank']
        cfg.tbfm.module.basis_residual_rank = hyperparams['basis_residual_rank']
    if 'residual_mlp_hidden' in hyperparams:
        cfg.meta.residual_mlp_hidden = hyperparams['residual_mlp_hidden']
        cfg.tbfm.module.residual_mlp_hidden = hyperparams['residual_mlp_hidden']
    if 'embed_dim_stim' in hyperparams:
        cfg.tbfm.module.embed_dim_stim = hyperparams['embed_dim_stim']
    OmegaConf.set_struct(cfg, True)

    # Workaround for should_warm_start
    OmegaConf.set_struct(cfg, False)
    if 'should_warm_start' not in cfg:
        if 'ae' in cfg and 'should_warm_start' in cfg.ae:
            cfg.should_warm_start = cfg.ae.should_warm_start
        else:
            cfg.should_warm_start = True
    OmegaConf.set_struct(cfg, True)

    # Load a small dataset just to build the model
    print("Loading data for model initialization...")
    adapt_sessions = metadata['adapt_session_ids']
    d, _ = multisession.load_stim_batched(
        batch_size=7500,
        window_size=185,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=adapt_sessions[:1],  # Just load one session for init
        num_held_out_sessions=0,
    )
    data_train, _ = d.train_test_split(5000, test_cut=2500)

    # Build model from config (without loading base model)
    print("Building model from config...")
    ms_model = multisession.build_from_cfg(
        cfg,
        data_train,
        base_model_path=None,  # Don't load base model
        device=device,
    )

    # Load the adapted state
    print("Loading adapted model state...")
    ms_model.load_state_dict(model_state, strict=False)  # Use strict=False to handle minor mismatches
    ms_model.eval()

    # Analyze each session
    print("\n" + "="*80)
    print("ANALYZING SESSIONS")
    print("="*80)

    results = {}
    for session_id in adapt_sessions:
        try:
            r2_overall, r2_per_channel = visualize_session_fit(session_id, ms_model, embeddings, device)
            results[session_id] = {'r2_overall': r2_overall, 'r2_per_channel': r2_per_channel}
        except Exception as e:
            print(f"Error analyzing {session_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for session_id, res in results.items():
        print(f"{session_id}: R²={res['r2_overall']:.4f}")

    print("\nVisualization complete! Check the generated PNG files.")


if __name__ == "__main__":
    main()
