#!/usr/bin/env python3
"""
Visualize model predictions on bad sessions to understand failure modes.
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from tbfm import multisession

# Constants
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")
BASE_MODEL_DIR = Path("/home/danmuir/GitHub/py-tbfm/session_count_sweep_20251203_222740/1kx25_maml")
ADAPTED_MODEL_DIR = Path("/home/danmuir/GitHub/py-tbfm/session_count_sweep_20251203_222740/tta_results/adapted_models/1kx25_maml_support5000_maml")

# Session to analyze
# Worst: SESSION_ID = "MonkeyJ_20160429_Session3_S1"  # R² = 0.129
# Best: SESSION_ID = "MonkeyG_20150917_Session3_S1"  # R² = 0.648
SESSION_ID = "MonkeyG_20150917_Session3_S1"  # Best performing session


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def main():
    print("="*80)
    print(f"VISUALIZING PREDICTIONS ON: {SESSION_ID}")
    print("="*80)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load metadata to verify session
    metadata = torch.load(ADAPTED_MODEL_DIR / 'metadata.torch')
    print("Adapted model metadata:")
    print(f"  Sessions: {metadata['adapt_session_ids']}")
    print(f"  Overall R²: {metadata['final_r2']:.4f}\n")

    if SESSION_ID not in metadata['adapt_session_ids']:
        print(f"Error: {SESSION_ID} not in adapted sessions!")
        return

    # Load configuration (use default config, will override with hyperparameters)
    print("Loading configuration...")
    config_dir = Path("/home/danmuir/GitHub/py-tbfm/conf")
    with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        cfg = compose(config_name="config")

    # Load and apply hyperparameters from base model
    print("Loading hyperparameters...")
    hyperparams = torch.load(BASE_MODEL_DIR / "hyperparameters.torch")

    OmegaConf.set_struct(cfg, False)
    cfg.latent_dim = hyperparams['latent_dim']
    cfg.ae.module.latent_dim = hyperparams['latent_dim']
    cfg.tbfm.module.in_dim = hyperparams['latent_dim']
    cfg.tbfm.module.num_bases = hyperparams['num_bases']
    cfg.meta.basis_residual_rank = hyperparams['basis_residual_rank']
    cfg.tbfm.module.basis_residual_rank = hyperparams['basis_residual_rank']
    cfg.meta.residual_mlp_hidden = hyperparams['residual_mlp_hidden']
    cfg.tbfm.module.residual_mlp_hidden = hyperparams['residual_mlp_hidden']
    cfg.tbfm.module.embed_dim_stim = hyperparams['embed_dim_stim']
    cfg.tbfm.module.is_basis_residual = hyperparams['is_basis_residual']  # Critical!
    cfg.should_warm_start = True
    OmegaConf.set_struct(cfg, True)

    # Load session data
    print(f"\nLoading session data for {SESSION_ID}...")
    d, _ = multisession.load_stim_batched(
        batch_size=500,  # Smaller batch for visualization
        window_size=184,  # From config
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=[SESSION_ID],
        num_held_out_sessions=0,
    )
    data_train, data_test = d.train_test_split(5000, test_cut=2500)

    # Build model from base
    print("Building model from base...")
    base_model_file = BASE_MODEL_DIR / "model.torch"
    ms_model = multisession.build_from_cfg(
        cfg,
        data_train,
        base_model_path=str(base_model_file),
        device=device,
    )

    # Load adapted weights
    print("Loading adapted model weights...")
    adapted_state = torch.load(ADAPTED_MODEL_DIR / 'model_adapted.torch', map_location=device)
    embeddings = torch.load(ADAPTED_MODEL_DIR / 'embeddings_stim_adapted.torch', map_location=device)

    # Load only the TBFM weights (not normalizers/AE which are frozen)
    ms_model.model.load_state_dict(adapted_state)
    ms_model.eval()

    print("Model loaded successfully!\n")

    # Get test data - use the data_test loader directly
    print("Getting test predictions...")
    test_batch = next(iter(data_test))

    # Load rest embeddings for the session
    print("Loading rest embeddings...")
    embeddings_rest = multisession.load_rest_embeddings([SESSION_ID], device=device)

    # Get stim embedding (from adapted model)
    session_emb_stim = embeddings[SESSION_ID].to(device)
    embeddings_stim_dict = {SESSION_ID: session_emb_stim}

    # Get predictions
    with torch.no_grad():
        # Move batch to device
        batch_device = {}
        for sid, data in test_batch.items():
            batch_device[sid] = tuple(d.to(device) for d in data)

        # Call forward with both embedding types
        y_pred_dict = ms_model(
            batch_device,
            embeddings_rest=embeddings_rest,
            embeddings_stim=embeddings_stim_dict
        )
        y_pred = y_pred_dict[SESSION_ID]

    # Get ground truth and normalize it to match predictions
    session_data = test_batch[SESSION_ID]
    y_true_raw = session_data[2].to(device)  # Target is element 2
    # Normalize the targets to match the prediction space
    y_true = ms_model.normalize({SESSION_ID: y_true_raw})[SESSION_ID]
    batch_size = y_true.shape[0]

    print(f"  Batch size: {batch_size}")
    print(f"  Prediction shape: {y_pred.shape}")
    print(f"  Target shape: {y_true.shape}")

    # Compute R²
    r2 = compute_r2(y_true, y_pred)
    print(f"\nR² score: {r2:.4f}")

    # Move to CPU for plotting
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_context_np = batch_device[SESSION_ID][0].cpu().numpy()  # Context from batch

    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'{SESSION_ID} - Model Predictions (R² = {r2:.4f})', fontsize=16, fontweight='bold')

    # Plot 1: Sample predictions for a few channels
    ax1 = fig.add_subplot(gs[0, :])
    n_samples_plot = 3
    n_channels_plot = min(5, y_true_np.shape[2])

    for i in range(n_samples_plot):
        for ch in range(n_channels_plot):
            offset = ch * 0.003  # Offset for visibility
            ax1.plot(y_true_np[i, :, ch] + offset, alpha=0.6,
                    color=f'C{ch}', linestyle='-', linewidth=1.5,
                    label=f'True Ch{ch}' if i == 0 else '')
            ax1.plot(y_pred_np[i, :, ch] + offset, alpha=0.6,
                    color=f'C{ch}', linestyle='--', linewidth=1.5,
                    label=f'Pred Ch{ch}' if i == 0 else '')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Neural Activity (offset for visibility)')
    ax1.set_title(f'Sample Predictions ({n_samples_plot} trials, {n_channels_plot} channels)')
    ax1.legend(ncol=n_channels_plot*2, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction scatter
    ax2 = fig.add_subplot(gs[1, 0])
    y_true_flat = y_true_np.flatten()
    y_pred_flat = y_pred_np.flatten()
    # Subsample for speed
    n_points = min(10000, len(y_true_flat))
    indices = np.random.choice(len(y_true_flat), n_points, replace=False)
    ax2.scatter(y_true_flat[indices], y_pred_flat[indices], alpha=0.1, s=1)
    lims = [min(y_true_flat.min(), y_pred_flat.min()),
            max(y_true_flat.max(), y_pred_flat.max())]
    ax2.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('True Value')
    ax2.set_ylabel('Predicted Value')
    ax2.set_title('Prediction Scatter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Residuals over time
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = y_true_np - y_pred_np
    residuals_mean = residuals.mean(axis=(0, 2))  # Average over batch and channels
    residuals_std = residuals.std(axis=(0, 2))
    time_steps = np.arange(len(residuals_mean))
    ax3.plot(time_steps, residuals_mean, 'b-', linewidth=2, label='Mean residual')
    ax3.fill_between(time_steps,
                      residuals_mean - residuals_std,
                      residuals_mean + residuals_std,
                      alpha=0.3, label='±1 std')
    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Residual (True - Pred)')
    ax3.set_title('Residuals Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residual distribution
    ax4 = fig.add_subplot(gs[1, 2])
    residuals_flat = residuals.flatten()
    ax4.hist(residuals_flat, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax4.axvline(residuals_flat.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {residuals_flat.mean():.4f}')
    ax4.set_xlabel('Residual')
    ax4.set_ylabel('Count')
    ax4.set_title('Residual Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5-6: Worst and best channels
    r2_per_channel = []
    for ch in range(y_true_np.shape[2]):
        r2_ch = compute_r2(
            torch.from_numpy(y_true_np[:, :, ch]),
            torch.from_numpy(y_pred_np[:, :, ch])
        )
        r2_per_channel.append(r2_ch)
    r2_per_channel = np.array(r2_per_channel)

    worst_ch = np.argmin(r2_per_channel)
    best_ch = np.argmax(r2_per_channel)

    # Worst channel
    ax5 = fig.add_subplot(gs[2, 0])
    for i in range(min(5, batch_size)):
        ax5.plot(y_true_np[i, :, worst_ch], alpha=0.6, color='blue',
                label='True' if i == 0 else '')
        ax5.plot(y_pred_np[i, :, worst_ch], alpha=0.6, color='red',
                linestyle='--', label='Pred' if i == 0 else '')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Activity')
    ax5.set_title(f'Worst Channel (ch={worst_ch}, R²={r2_per_channel[worst_ch]:.3f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Best channel
    ax6 = fig.add_subplot(gs[2, 1])
    for i in range(min(5, batch_size)):
        ax6.plot(y_true_np[i, :, best_ch], alpha=0.6, color='blue',
                label='True' if i == 0 else '')
        ax6.plot(y_pred_np[i, :, best_ch], alpha=0.6, color='red',
                linestyle='--', label='Pred' if i == 0 else '')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Activity')
    ax6.set_title(f'Best Channel (ch={best_ch}, R²={r2_per_channel[best_ch]:.3f})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # R² per channel
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.bar(range(len(r2_per_channel)), r2_per_channel, alpha=0.7)
    ax7.axhline(r2, color='red', linestyle='--', label=f'Overall: {r2:.3f}')
    ax7.set_xlabel('Channel')
    ax7.set_ylabel('R² Score')
    ax7.set_title('R² by Channel')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(f"/home/danmuir/GitHub/py-tbfm/predictions_{SESSION_ID}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    # Print statistics
    print("\n" + "="*80)
    print("PREDICTION STATISTICS")
    print("="*80)
    print(f"Overall R²: {r2:.4f}")
    print(f"Mean absolute error: {np.abs(residuals_flat).mean():.6f}")
    print(f"RMSE: {np.sqrt((residuals_flat**2).mean()):.6f}")
    print(f"R² range across channels: [{r2_per_channel.min():.4f}, {r2_per_channel.max():.4f}]")
    print(f"Mean R² across channels: {r2_per_channel.mean():.4f}")


if __name__ == "__main__":
    main()
