#!/usr/bin/env python3
"""
State Dependency Visualizer

Generates state dependency graphs for multiple sessions using either:
1. Test-time adaptation (TTA) with a pretrained multisession model, or
2. Vanilla single-session TBFM training from scratch (--use-vanilla)

Results are saved in organized folders with subfolders for each session.

Multi-GPU Support:
- Use --gpus to specify multiple GPU IDs for parallel processing
- Sessions are distributed across GPUs in round-robin fashion
- Example: --gpus 0 1 2 3 (processes up to 4 sessions in parallel)
- Resume support: --force-overwrite to reprocess, otherwise skips completed sessions
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import multiprocessing as mp
from functools import partial

from tbfm import multisession, test, tbfm as tbfm_module, utils
from torcheval.metrics.functional import r2_score


# Constants
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")

# Default sessions for vanilla model evaluation (40 sessions)
DEFAULT_VANILLA_SESSIONS = [
    "MonkeyG_20150914_Session1_S1",
    "MonkeyG_20150914_Session3_S1",
    "MonkeyG_20150915_Session2_S1",
    "MonkeyG_20150915_Session3_S1",
    "MonkeyG_20150915_Session4_S1",
    "MonkeyG_20150915_Session5_S1",
    "MonkeyG_20150916_Session4_S1",
    "MonkeyG_20150917_Session1_M1",
    "MonkeyG_20150917_Session1_S1",
    "MonkeyG_20150917_Session2_M1",
    "MonkeyG_20150917_Session2_S1",
    "MonkeyG_20150917_Session3_M1",
    "MonkeyG_20150917_Session3_S1",
    "MonkeyG_20150918_Session1_M1",
    "MonkeyG_20150918_Session1_S1",
    "MonkeyG_20150921_Session3_S1",
    "MonkeyG_20150921_Session5_S1",
    "MonkeyG_20150922_Session1_S1",
    "MonkeyG_20150922_Session2_S1",
    "MonkeyG_20150922_Session3_S1",
    "MonkeyG_20150925_Session1_S1",
    "MonkeyG_20150925_Session2_S1",
    "MonkeyJ_20160426_Session1_S1",
    "MonkeyJ_20160426_Session2_S1",
    "MonkeyJ_20160426_Session3_S1",
    "MonkeyJ_20160428_Session2_S1",
    "MonkeyJ_20160428_Session3_S1",
    "MonkeyJ_20160429_Session1_S1",
    "MonkeyJ_20160429_Session3_S1",
    "MonkeyJ_20160502_Session1_S1",
    "MonkeyJ_20160624_Session3_S1",
    "MonkeyJ_20160624_Session4_S1",
    "MonkeyJ_20160625_Session4_S1",
    "MonkeyJ_20160625_Session5_S1",
    "MonkeyJ_20160627_Session1_S1",
    "MonkeyJ_20160627_Session2_S1",
    "MonkeyJ_20160630_Session1_S1",
    "MonkeyJ_20160630_Session3_S1",
    "MonkeyJ_20160702_Session2_S1",
    "MonkeyJ_20160702_Session4_S1",
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate state dependency visualizations for multiple sessions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=False,
        help="Path to model directory containing model.torch and hyperparameters.torch (not required for vanilla mode)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=str,
        default=None,
        help="Session IDs to visualize (if not provided, uses all available test sessions)"
    )
    parser.add_argument(
        "--support-samples",
        type=int,
        default=2500,
        help="Number of samples to use for TTA"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2500,
        help="Number of samples to use for testing"
    )
    parser.add_argument(
        "--tta-epochs",
        type=int,
        default=7001,
        help="Number of epochs for TTA"
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=5,
        help="Number of bins for state discretization"
    )
    parser.add_argument(
        "--runway-length",
        type=int,
        default=20,
        help="Runway length (number of initial timesteps used as input)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=184,
        help="Total trial window size"
    )
    parser.add_argument(
        "--batch-size-per-session",
        type=int,
        default=7500,
        help="Batch size per session for data loading"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: state_dependency_graphs_<timestamp>)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("./conf"),
        help="Hydra configuration directory"
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA device ID to use (deprecated: use --gpus instead)"
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=str,
        default=None,
        help="GPU IDs to use for parallel processing (e.g., --gpus 0 1 2 3). If multiple GPUs are provided, sessions will be processed in parallel."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    parser.add_argument(
        "--use-vanilla",
        action="store_true",
        help="Use vanilla model without TTA (test-time adaptation)"
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force reprocessing of sessions even if already complete"
    )

    return parser.parse_args()


def is_session_complete(output_dir, session_id):
    """
    Check if a session has already been processed.

    Args:
        output_dir: Output directory
        session_id: Session identifier

    Returns:
        True if session processing is complete, False otherwise
    """
    session_dir = output_dir / session_id
    summary_path = session_dir / "summary.txt"
    return summary_path.exists()


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def evaluate_model(session_id, ms_model, embeddings_stim, embeddings_rest, data_loader, device):
    """Evaluate model on a data loader and return R² and predictions."""
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch_device = {}
            for sid, data in batch.items():
                batch_device[sid] = tuple(d.to(device) for d in data)

            # Get predictions
            y_pred_dict = ms_model(
                batch_device,
                embeddings_rest=embeddings_rest,
                embeddings_stim={session_id: embeddings_stim[session_id]}
            )
            y_pred = y_pred_dict[session_id]

            # Get ground truth
            y_true_raw = batch[session_id][2].to(device)
            y_true = ms_model.normalize({session_id: y_true_raw})[session_id]

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

    # Concatenate all batches
    y_true = torch.cat(all_y_true, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)

    # Compute R²
    r2 = compute_r2(y_true, y_pred)

    return r2, y_true, y_pred


def generate_state_dependency_graphs(
    session_id,
    y_true,
    y_pred,
    num_channels,
    runway_length,
    bin_count,
    output_dir,
    prefix="test"
):
    """
    Generate and save state dependency graphs for all channels.

    Args:
        session_id: Session identifier
        y_true: Ground truth predictions (full trials, including runway)
        y_pred: Model predictions (only post-runway portion)
        num_channels: Number of channels to visualize
        runway_length: Length of runway portion
        bin_count: Number of bins for state discretization
        output_dir: Directory to save graphs
        prefix: Prefix for filenames (e.g., "train" or "test")
    """
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    for ch in range(num_channels):
        title = f"{prefix.capitalize()} set - {session_id} (Channel {ch})"
        savepath = session_dir / f"{prefix}_channel_{ch:02d}.png"

        test.graph_state_dependency(
            y_true,
            y_pred,
            ch=ch,
            runway_length=runway_length,
            bin_count=bin_count,
            title=title,
            savepath=str(savepath)
        )
        plt.close('all')  # Clean up memory


def train_vanilla_tbfm_for_session(
    cfg,
    session_id,
    data_train,
    data_test,
    support_size,
    epochs,
    device,
    quiet=False
):
    """
    Train a vanilla (single-session) TBFM model on the support set for one session.

    Args:
        cfg: Config object
        session_id: Session identifier
        data_train: Training data loader
        data_test: Test data loader
        support_size: Number of samples to use for training
        epochs: Number of training epochs
        device: Device to train on
        quiet: If True, suppress progress messages

    Returns:
        Tuple of (trained_model, train_r2, test_r2, y_true_train, y_pred_train, y_true_test, y_pred_test)
    """
    if not quiet:
        print(f"Training vanilla TBFM for {session_id}...")

    device_obj = torch.device(device) if isinstance(device, str) else device

    # Materialize support set
    _, data_support = utils.iter_loader(iter(data_train), data_train, device=str(device_obj))

    # Filter support data to requested size
    data_support, _ = multisession.split_support_query_sessions(data_support, support_size)

    # Get data for this session
    runway_train, stiminds_train, y_train = data_support[session_id]

    # Get dimensions
    in_dim = runway_train.shape[-1]
    runway_len = runway_train.shape[1]
    trial_len = y_train.shape[1]
    stim_dim = stiminds_train.shape[-1] + 1  # +1 to account for clock dimension

    # Get hyperparameters from config
    num_bases = 12  # Vanilla uses fewer bases
    latent_dim = in_dim  # Identity mapping
    basis_depth = cfg.tbfm.module.basis_depth
    lambda_fro = cfg.tbfm.training.lambda_fro

    # Create vanilla TBFM model
    _tbfm = tbfm_module.TBFM(
        in_dim=in_dim,
        stimdim=stim_dim,
        runway=runway_len,
        num_bases=num_bases,
        trial_len=trial_len,
        batchy=y_train,
        latent_dim=latent_dim,
        basis_depth=basis_depth,
        zscore=True,
        use_meta_learning=False,
        device=device_obj,
    )

    # Normalize training data
    y_train_norm = _tbfm.normalize(y_train)

    # Get optimizer
    optim = _tbfm.get_optim(lr=cfg.tbfm.training.optim.lr_head)

    # Training loop
    _tbfm.train()
    for epoch in range(epochs):
        optim.zero_grad()
        y_pred = _tbfm(runway_train, stiminds_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train_norm)

        if lambda_fro > 0:
            loss_reg = _tbfm.get_weighting_reg()
            loss = loss + lambda_fro * loss_reg

        loss.backward()
        optim.step()

        if not quiet and epoch % 1000 == 0:
            print(f"  Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate on training set
    _tbfm.eval()
    with torch.no_grad():
        y_pred_train = _tbfm(runway_train, stiminds_train)
        train_r2 = r2_score(y_pred_train.flatten(), y_train_norm.flatten()).item()

    # Evaluate on test set
    with torch.no_grad():
        test_r2_acc = 0.0
        test_batch_count = 0
        all_y_true_test = []
        all_y_pred_test = []

        for test_batch_dict in data_test:
            if session_id not in test_batch_dict:
                continue

            runway_test, stiminds_test, y_test = test_batch_dict[session_id]
            runway_test = runway_test.to(device_obj)
            stiminds_test = stiminds_test.to(device_obj)
            y_test = y_test.to(device_obj)

            y_test_norm = _tbfm.normalize(y_test)
            y_pred_test = _tbfm(runway_test, stiminds_test)

            all_y_true_test.append(y_test_norm)
            all_y_pred_test.append(y_pred_test)

            test_r2 = r2_score(y_pred_test.flatten(), y_test_norm.flatten())
            test_r2_acc += test_r2.item()
            test_batch_count += 1

        test_r2_final = test_r2_acc / test_batch_count if test_batch_count > 0 else 0.0

        # Concatenate all test predictions
        y_true_test = torch.cat(all_y_true_test, dim=0) if all_y_true_test else y_train_norm
        y_pred_test = torch.cat(all_y_pred_test, dim=0) if all_y_pred_test else y_pred_train

    if not quiet:
        print(f"  Vanilla TBFM training complete. Train R²: {train_r2:.4f}, Test R²: {test_r2_final:.4f}")

    return _tbfm, train_r2, test_r2_final, y_train_norm, y_pred_train, y_true_test, y_pred_test


def save_model_state(ms_model, embeddings_stim, output_dir, session_id):
    """
    Save adapted model state after TTA.

    During TTA, fresh AEs are created and optimized for each session while the base
    TBFM model remains frozen. We need to save the adapted AE and normalizer states.

    Args:
        ms_model: Multisession model
        embeddings_stim: Stimulus embeddings
        output_dir: Output directory
        session_id: Session identifier
    """
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save adapted AE state for this session (optimized during TTA)
    ae_path = session_dir / "adapted_ae.torch"
    if hasattr(ms_model.ae, 'instances') and session_id in ms_model.ae.instances:
        ae_instance = ms_model.ae.instances[session_id]
        torch.save(ae_instance.state_dict(), ae_path)
        print(f"  Saved adapted AE to {ae_path}")

    # Save normalizer state for this session
    norm_path = session_dir / "normalizer.torch"
    if hasattr(ms_model.norms, 'instances') and session_id in ms_model.norms.instances:
        norm_instance = ms_model.norms.instances[session_id]
        torch.save(norm_instance.state_dict(), norm_path)
        print(f"  Saved normalizer to {norm_path}")

    # Save stimulus embeddings
    embeddings_path = session_dir / "embeddings_stim.torch"
    torch.save(embeddings_stim, embeddings_path)
    print(f"  Saved embeddings to {embeddings_path}")


def save_vanilla_model_state(vanilla_model, output_dir, session_id):
    """
    Save vanilla model state after training.

    Args:
        vanilla_model: Vanilla TBFM model
        output_dir: Output directory
        session_id: Session identifier
    """
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save vanilla model state
    model_path = session_dir / "vanilla_model.torch"
    torch.save(vanilla_model.state_dict(), model_path)

    print(f"  Saved vanilla model to {model_path}")


def process_session_worker(
    session_id,
    gpu_id,
    args,
    cfg,
    held_in_sessions,
    base_model_file,
    prev_session_id=None
):
    """
    Worker function to process a single session on a specific GPU.

    Args:
        session_id: Session identifier to process
        gpu_id: GPU device ID to use
        args: Parsed command-line arguments
        cfg: Hydra configuration
        held_in_sessions: List of held-in session IDs
        base_model_file: Path to base model file (for TTA mode)
        prev_session_id: Previous session ID for loading adapted parameters (optional)

    Returns:
        Dictionary with processing results
    """
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    try:
        print(f"\n{'='*80}")
        print(f"Processing session: {session_id} on device: {device}")
        print(f"{'='*80}")

        # Check if session is already complete
        if not args.force_overwrite and is_session_complete(args.output_dir, session_id):
            print(f"✓ Session {session_id} already complete - skipping")
            return {
                'session_id': session_id,
                'status': 'skipped',
                'message': 'Already complete'
            }

        if is_session_complete(args.output_dir, session_id):
            print(f"⚠ Session {session_id} already exists but will be overwritten (--force-overwrite enabled)")

        # Load session data
        print(f"Loading data for {session_id}...")
        batch_size = args.batch_size_per_session * len([session_id])

        d, _ = multisession.load_stim_batched(
            batch_size=batch_size,
            window_size=args.window_size,
            session_subdir="torchraw",
            data_dir=DATA_DIR,
            held_in_session_ids=[session_id],
            num_held_out_sessions=0,
        )
        data_train, data_test = d.train_test_split(args.support_samples, test_cut=args.test_samples)

        # Get number of channels from data
        sample_batch = next(iter(data_train))
        num_channels = sample_batch[session_id][0].shape[-1]
        print(f"  Number of channels: {num_channels}")

        if args.use_vanilla:
            # Vanilla mode: Train a single-session TBFM model
            vanilla_model, r2_train, r2_test, y_true_train, y_pred_train, y_true_test, y_pred_test = train_vanilla_tbfm_for_session(
                cfg,
                session_id,
                data_train,
                data_test,
                args.support_samples,
                args.tta_epochs,
                device,
                quiet=args.quiet
            )

            print(f"\nVanilla model results:")
            print(f"  Train R²: {r2_train:.4f}")
            print(f"  Test R²: {r2_test:.4f}")

            # Create tta_results dict for consistency
            tta_results = {'final_test_r2': r2_test}
        else:
            # TTA mode: Use multisession model with adaptation
            # Load rest embeddings
            embeddings_rest = multisession.load_rest_embeddings(held_in_sessions, device=device)
            session_embeddings_rest = multisession.load_rest_embeddings([session_id], device=device)
            embeddings_rest.update(session_embeddings_rest)

            # Build model
            print("Building model...")
            ms_model = multisession.build_from_cfg(
                cfg,
                data_train,
                base_model_path=str(base_model_file),
                device=device,
            )

            # Load previously adapted parameters if available
            if prev_session_id is not None:
                prev_model_path = args.output_dir / prev_session_id / "adapted_model.torch"
                if prev_model_path.exists():
                    print(f"Loading adapted parameters from previous session ({prev_session_id})...")
                    prev_state = torch.load(prev_model_path, map_location=device)
                    current_state = ms_model.state_dict()

                    # Update only the AE parameters
                    for key in prev_state.keys():
                        if 'ae.' in key and key in current_state:
                            current_state[key] = prev_state[key]

                    ms_model.load_state_dict(current_state, strict=False)
                    print("  Previous adaptations restored")

            ms_model.eval()

            # Run TTA
            print(f"\nRunning TTA with {args.support_samples} support samples...")
            embeddings_stim_adapted, tta_results = multisession.test_time_adaptation(
                cfg,
                ms_model,
                embeddings_rest,
                data_train,
                epochs=args.tta_epochs,
                data_test=data_test,
                ae_warm_start=True,
                adapt_ae=True,
                support_size=args.support_samples,
                coadapt_embeddings=False,
                quiet=args.quiet,
            )

            print(f"TTA complete! Final test R²: {tta_results['final_test_r2']:.4f}")

            # Evaluate on train set
            print("\nEvaluating on train set...")
            r2_train, y_true_train, y_pred_train = evaluate_model(
                session_id, ms_model, embeddings_stim_adapted, embeddings_rest, data_train, device
            )
            print(f"  Train R²: {r2_train:.4f}")

            # Evaluate on test set
            print("Evaluating on test set...")
            r2_test, y_true_test, y_pred_test = evaluate_model(
                session_id, ms_model, embeddings_stim_adapted, embeddings_rest, data_test, device
            )
            print(f"  Test R²: {r2_test:.4f}")

        # Get full trials for state dependency graphs
        print("\nReconstructing full trial data...")
        train_batch = next(iter(data_train))
        train_runway = train_batch[session_id][0].to(device)
        train_stim = train_batch[session_id][1].to(device)
        train_y_raw = train_batch[session_id][2].to(device)
        full_train_trials = torch.cat([train_runway, train_y_raw], dim=1)

        test_batch = next(iter(data_test))
        test_runway = test_batch[session_id][0].to(device)
        test_stim = test_batch[session_id][1].to(device)
        test_y_raw = test_batch[session_id][2].to(device)
        full_test_trials = torch.cat([test_runway, test_y_raw], dim=1)

        if args.use_vanilla:
            full_y_train = vanilla_model.normalize(full_train_trials)
            full_y_test = vanilla_model.normalize(full_test_trials)
        else:
            full_y_train = ms_model.normalize({session_id: full_train_trials})[session_id]
            full_y_test = ms_model.normalize({session_id: full_test_trials})[session_id]

        # Generate state dependency graphs
        print("\nGenerating state dependency graphs...")
        print(f"  Train set: {num_channels} channels")
        generate_state_dependency_graphs(
            session_id,
            full_y_train,
            y_pred_train,
            num_channels,
            args.runway_length,
            args.bin_count,
            args.output_dir,
            prefix="train"
        )

        print(f"  Test set: {num_channels} channels")
        generate_state_dependency_graphs(
            session_id,
            full_y_test,
            y_pred_test,
            num_channels,
            args.runway_length,
            args.bin_count,
            args.output_dir,
            prefix="test"
        )

        # Save model
        print("\nSaving model...")
        if args.use_vanilla:
            save_vanilla_model_state(vanilla_model, args.output_dir, session_id)
        else:
            save_model_state(ms_model, embeddings_stim_adapted, args.output_dir, session_id)

        # Save summary statistics
        summary_path = args.output_dir / session_id / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Session: {session_id}\n")
            f.write(f"Mode: {'Vanilla (no TTA)' if args.use_vanilla else 'TTA (test-time adaptation)'}\n")
            f.write(f"Support samples: {args.support_samples}\n")
            f.write(f"Test samples: {args.test_samples}\n")
            f.write(f"Training epochs: {args.tta_epochs}\n")
            f.write(f"Number of channels: {num_channels}\n")
            f.write(f"GPU: {device}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Train R²: {r2_train:.6f}\n")
            f.write(f"  Test R²: {r2_test:.6f}\n")
            if not args.use_vanilla:
                f.write(f"  Final test R² (from TTA): {tta_results['final_test_r2']:.6f}\n")

        print(f"\n✓ Session {session_id} complete!")
        print(f"  Results saved to {args.output_dir / session_id}")

        return {
            'session_id': session_id,
            'status': 'success',
            'r2_train': r2_train,
            'r2_test': r2_test,
            'device': device
        }

    except Exception as e:
        print(f"\n✗ Error processing session {session_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'session_id': session_id,
            'status': 'error',
            'error': str(e)
        }


def main():
    args = parse_args()

    # Validate arguments
    if not args.use_vanilla and args.model_dir is None:
        raise ValueError("--model-dir is required when not using vanilla mode. Use --use-vanilla for vanilla mode.")

    # Determine which GPUs to use
    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        gpu_ids = [args.cuda_device]

    use_parallel = len(gpu_ids) > 1
    print(f"Using GPU(s): {', '.join(gpu_ids)}")
    if use_parallel:
        print(f"Parallel processing mode: {len(gpu_ids)} GPUs")
    else:
        print(f"Sequential processing mode: 1 GPU")

    # Create output directory
    if args.output_dir is None:
        if args.use_vanilla:
            args.output_dir = Path(f"state_dependency_results_vanilla_{args.support_samples}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = Path(f"state_dependency_graphs_{timestamp}_{args.support_samples}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'Vanilla (no TTA)' if args.use_vanilla else 'TTA (test-time adaptation)'}")

    # Load configuration
    print("\nLoading configuration...")
    with initialize_config_dir(config_dir=str(args.config_dir.absolute()), version_base=None):
        cfg = compose(config_name="config")

    # Load and apply hyperparameters (only for TTA mode)
    if not args.use_vanilla:
        print("Loading hyperparameters...")
        hyperparams = torch.load(args.model_dir / "hyperparameters.torch")

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
        cfg.tbfm.module.is_basis_residual = hyperparams['is_basis_residual']
        cfg.should_warm_start = True
        OmegaConf.set_struct(cfg, True)

        # Load held-in sessions (for rest embeddings)
        print("Loading held-in sessions...")
        hisi_path = args.model_dir / "hisi.torch"
        if not hisi_path.exists():
            hisi_path = args.model_dir / "hisi_nf_1.torch"
        held_in_sessions = torch.load(hisi_path)
        print(f"  Found {len(held_in_sessions)} held-in sessions")
    else:
        # In vanilla mode, we don't need held-in sessions
        held_in_sessions = []

    # Determine sessions to visualize
    if args.sessions is None:
        if args.use_vanilla:
            # In vanilla mode, use the default 40 sessions
            print("\nNo sessions specified, using default 40 sessions for vanilla mode...")
            sessions_to_process = DEFAULT_VANILLA_SESSIONS
            print(f"  Using {len(sessions_to_process)} default vanilla sessions")
        else:
            # Load all held-out sessions (not held-in sessions)
            print("\nNo sessions specified, loading all held-out sessions...")
            # gather_session_ids returns held-in and held-out session IDs
            _, held_out_session_ids = multisession.gather_session_ids(
                DATA_DIR,
                num_held_out_sessions=0,  # This means all non-held-in sessions
                held_in_session_ids=held_in_sessions
            )

            # Filter to only sessions with rest embeddings
            print("Filtering sessions with available rest embeddings...")
            sessions_to_process = []
            for session_id in held_out_session_ids:
                rest_embed_path = Path(DATA_DIR) / session_id / "embedding_rest" / "er.torch"
                if rest_embed_path.exists():
                    sessions_to_process.append(session_id)

            print(f"  Found {len(sessions_to_process)} held-out sessions with rest embeddings")
            print(f"  (out of {len(held_out_session_ids)} total held-out sessions)")
    else:
        sessions_to_process = args.sessions
        print(f"\nProcessing {len(sessions_to_process)} specified sessions")

    # Save session list before running TTA
    session_list_path = args.output_dir / "session_list.txt"
    with open(session_list_path, 'w') as f:
        f.write("Sessions to process:\n")
        for i, session_id in enumerate(sessions_to_process):
            f.write(f"{i+1}. {session_id}\n")
        f.write(f"\nTotal: {len(sessions_to_process)} sessions\n")
        f.write(f"Held-in sessions: {len(held_in_sessions)}\n")
    print(f"Saved session list to {session_list_path}")

    # Load base model file path (only needed for TTA mode)
    if not args.use_vanilla:
        print("\nLoading base model...")
        base_model_file = args.model_dir / "model.torch"
    else:
        base_model_file = None

    # Check for already completed sessions
    if not args.force_overwrite:
        completed_sessions = [s for s in sessions_to_process if is_session_complete(args.output_dir, s)]
        if completed_sessions:
            print(f"\nFound {len(completed_sessions)} already completed sessions (will skip)")
            print("Use --force-overwrite to reprocess completed sessions")

    # Process sessions
    print(f"\n{'='*80}")
    print(f"Starting processing of {len(sessions_to_process)} sessions")
    print(f"{'='*80}")

    if use_parallel:
        # Parallel processing with multiprocessing
        print("\nStarting parallel processing...")
        print("Note: In parallel mode, sessions are processed independently without loading adapted parameters from previous sessions.")

        # Set start method for multiprocessing (only if not already set)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set

        # Prepare tasks: assign GPUs in round-robin fashion
        # In parallel mode, we don't use prev_session_id since sessions are processed independently
        tasks = []
        for idx, session_id in enumerate(sessions_to_process):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            tasks.append((session_id, gpu_id, args, cfg, held_in_sessions, base_model_file, None))

        # Process in parallel
        with mp.Pool(processes=len(gpu_ids)) as pool:
            results = pool.starmap(process_session_worker, tasks)

    else:
        # Sequential processing (original behavior)
        print("\nStarting sequential processing...")
        results = []
        gpu_id = gpu_ids[0]

        for idx, session_id in enumerate(sessions_to_process):
            prev_session_id = sessions_to_process[idx - 1] if idx > 0 else None
            result = process_session_worker(
                session_id=session_id,
                gpu_id=gpu_id,
                args=args,
                cfg=cfg,
                held_in_sessions=held_in_sessions,
                base_model_file=base_model_file,
                prev_session_id=prev_session_id
            )
            results.append(result)

    # Process results
    sessions_processed = sum(1 for r in results if r['status'] == 'success')
    sessions_skipped = sum(1 for r in results if r['status'] == 'skipped')
    sessions_failed = sum(1 for r in results if r['status'] == 'error')

    # Print detailed results
    if sessions_failed > 0:
        print(f"\n{'='*80}")
        print("Failed sessions:")
        print(f"{'='*80}")
        for r in results:
            if r['status'] == 'error':
                print(f"  {r['session_id']}: {r.get('error', 'Unknown error')}")

    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"  Sessions processed: {sessions_processed}")
    print(f"  Sessions skipped (already complete): {sessions_skipped}")
    print(f"  Sessions failed: {sessions_failed}")
    print(f"  Total sessions: {len(sessions_to_process)}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
