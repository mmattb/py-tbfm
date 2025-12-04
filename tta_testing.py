#!/usr/bin/env python3
"""
Test-Time Adaptation (TTA) Evaluation Script

Evaluates TTA performance across multiple models, support sizes, and adaptation strategies.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tbfm import dataset, multisession, utils
import notifications


# Constants
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")
EMBEDDING_REST_SUBDIR = "embedding_rest"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test-Time Adaptation Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="1",
        help="CUDA device ID to use (ignored if --use-multi-gpu is set)"
    )
    parser.add_argument(
        "--use-multi-gpu",
        action="store_true",
        help="Use all available GPUs for parallel processing"
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="Specific GPU IDs to use (e.g., --gpu-ids 0 1 2 3). If not specified, uses all available GPUs."
    )
    parser.add_argument(
        "--support-sizes",
        nargs="+",
        type=int,
        default=[50, 100, 250, 500, 1000, 2500, 5000],
        help="Support set sizes to evaluate"
    )
    parser.add_argument(
        "--max-adapt-sessions",
        type=int,
        default=1,
        help="Maximum number of adaptation sessions to use (ignored if --adapt-session is specified)"
    )
    parser.add_argument(
        "--adapt-session",
        type=str,
        default=None,
        help="Specific session ID to use for TTA adaptation (e.g., 'MonkeyG_20150914_Session1_S1')"
    )
    parser.add_argument(
        "--tta-epochs",
        type=int,
        default=7001,
        help="Number of epochs for TTA (outer steps)"
    )
    parser.add_argument(
        "--tta-inner-steps",
        type=int,
        default=20,
        help="Number of inner steps for MAML strategy"
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
        default=Path("data/tta_sweep"),
        help="Directory for output files"
    )
    parser.add_argument(
        "--no-plot-display",
        action="store_true",
        help="Don't display plot interactively (only save)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("./conf"),
        help="Hydra configuration directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to evaluate (if not provided, uses all)"
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        type=str,
        help="Custom model paths in format 'name:path' (e.g., '100_5_inner:test/100_5_rr16_inner_ts1000')"
    )
    parser.add_argument(
        "--include-vanilla-tbfm",
        action="store_true",
        help="Include vanilla TBFM baseline trained on support set for comparison"
    )
    parser.add_argument(
        "--vanilla-tbfm-epochs",
        type=int,
        default=7001,
        help="Number of epochs for vanilla TBFM training"
    )
    parser.add_argument(
        "--include-fresh-tbfm",
        action="store_true",
        help="Include fresh TBFM (no multisession features) baseline for comparison"
    )
    parser.add_argument(
        "--fresh-tbfm-epochs",
        type=int,
        default=7001,
        help="Number of epochs for fresh TBFM training"
    )
    parser.add_argument(
        "--progressive-unfreezing-threshold",
        type=int,
        default=0,
        help="Support size threshold for enabling progressive unfreezing"
    )
    parser.add_argument(
        "--unfreeze-basis-weights",
        action="store_true",
        help="Enable supervised fine-tuning of basis weights at high support sizes"
    )
    parser.add_argument(
        "--unfreeze-bases",
        action="store_true",
        help="Enable supervised fine-tuning of basis generator at high support sizes"
    )
    parser.add_argument(
        "--basis-weight-lr",
        type=float,
        default=1e-5,
        help="Learning rate for basis weights when unfrozen"
    )
    parser.add_argument(
        "--bases-lr",
        type=float,
        default=1e-6,
        help="Learning rate for basis generator when unfrozen"
    )

    return parser.parse_args()


def setup_environment(cuda_device: str):
    """Configure CUDA environment variables."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    print(f"Using CUDA device: {cuda_device}")


def gpu_worker(
    gpu_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    cfg_dict: dict,
    model_paths_dict: dict,
    adapt_session_ids: list,
    window_size: int,
    batch_size_per_session: int,
    tta_epochs: int,
    tta_strategies: dict,
    output_dir: Path = None,
):
    """
    Worker process for processing TTA jobs on a specific GPU.
    
    Args:
        gpu_id: GPU device ID to use
        job_queue: Queue of jobs to process (model_key, support_size, strategy_key)
        result_queue: Queue to put results
        cfg_dict: Configuration dictionary
        model_paths_dict: Model paths dictionary
        adapt_session_ids: List of session IDs to load data for
        window_size: Window size for data loading
        batch_size_per_session: Batch size per session
        tta_epochs: Number of TTA epochs
        tta_strategies: TTA strategy configurations
        output_dir: Directory to save adapted models
    """
    # Set up this process's GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # After setting CUDA_VISIBLE_DEVICES, use device 0
    
    # Reconstruct config from dict
    cfg = OmegaConf.create(cfg_dict)
    
    # Workaround: ae.py expects cfg.should_warm_start but config might have it under cfg.ae.should_warm_start
    OmegaConf.set_struct(cfg, False)
    if 'should_warm_start' not in cfg:
        if 'ae' in cfg and 'should_warm_start' in cfg.ae:
            cfg.should_warm_start = cfg.ae.should_warm_start
        else:
            cfg.should_warm_start = True
    OmegaConf.set_struct(cfg, True)
    
    print(f"[GPU {gpu_id}] Worker started, loading data...")    # Load data in this worker process
    batch_size = batch_size_per_session * len(adapt_session_ids)
    d, _ = multisession.load_stim_batched(
        batch_size=batch_size,
        window_size=window_size,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=adapt_session_ids,
        num_held_out_sessions=0,
    )
    data_train, data_test = d.train_test_split(5000, test_cut=2500)
    
    # Load embeddings for all models
    embeddings_cache = {}
    for model_key in model_paths_dict.keys():
        # Load held-in sessions for this model
        model_path = Path(model_paths_dict[model_key])
        hisi_path = None
        for candidate in ["hisi_nf_1.torch", "hisi.torch"]:
            candidate_path = model_path / candidate
            if candidate_path.exists():
                hisi_path = candidate_path
                break
        
        if hisi_path is None:
            print(f"[GPU {gpu_id}] Warning: No hisi file found for {model_key}")
            continue
        
        held_in_sessions = torch.load(hisi_path)
        embeds = multisession.load_rest_embeddings(held_in_sessions, device=device)
        embeds.update(multisession.load_rest_embeddings(adapt_session_ids, device=device))
        embeddings_cache[model_key] = embeds
    
    print(f"[GPU {gpu_id}] Data loaded, ready to process jobs")
    
    while True:
        try:
            # Get job from queue (timeout to check if we should exit)
            job = job_queue.get(timeout=1)
            
            if job is None:  # Poison pill to signal worker to exit
                print(f"[GPU {gpu_id}] Worker received exit signal")
                break
            
            model_key, support_size, strategy_key = job
            strategy_cfg = tta_strategies[strategy_key]
            
            print(f"[GPU {gpu_id}] Processing: Model={model_key}, Support={support_size}, Strategy={strategy_key}")
            
            # Create progress notification for this TTA run
            job_id = None
            try:
                job_id = notifications.create_progress_notification(
                    job_id=f"tta_{model_key}_{support_size}_{strategy_key}_gpu{gpu_id}",
                    title=f"TTA: {model_key}",
                    initial_message=f"Starting {strategy_key} with support size {support_size}...\nGPU: {gpu_id}"
                )
            except Exception as e:
                print(f"[GPU {gpu_id}] Failed to create progress notification: {e}")
            
            try:
                # Clone config and apply model-specific hyperparameters
                cfg_eval = clone_cfg_for_eval(cfg)
                
                # Reconstruct model path
                model_path = Path(model_paths_dict[model_key])
                
                # Load and apply model-specific hyperparameters
                params = load_model_hyperparameters(model_path)
                print(f"[GPU {gpu_id}] Parsed params for {model_key}: {params}")
                
                if params:
                    # Need to disable struct mode to modify config
                    OmegaConf.set_struct(cfg_eval, False)
                    
                    if 'latent_dim' in params and params['latent_dim'] is not None:
                        cfg_eval.latent_dim = params['latent_dim']
                        # Also update ae.module.latent_dim since interpolation may be resolved
                        cfg_eval.ae.module.latent_dim = params['latent_dim']
                        # Also update tbfm.module.in_dim which uses ${latent_dim}
                        cfg_eval.tbfm.module.in_dim = params['latent_dim']
                    if 'num_bases' in params and params['num_bases'] is not None:
                        cfg_eval.tbfm.module.num_bases = params['num_bases']
                    if 'basis_residual_rank' in params and params['basis_residual_rank'] is not None:
                        cfg_eval.meta.basis_residual_rank = params['basis_residual_rank']
                        # Also update tbfm.module.basis_residual_rank which uses ${meta.basis_residual_rank}
                        cfg_eval.tbfm.module.basis_residual_rank = params['basis_residual_rank']
                    if 'residual_mlp_hidden' in params and params['residual_mlp_hidden'] is not None:
                        cfg_eval.meta.residual_mlp_hidden = params['residual_mlp_hidden']
                        # Also update tbfm.module.residual_mlp_hidden which uses ${meta.residual_mlp_hidden}
                        cfg_eval.tbfm.module.residual_mlp_hidden = params['residual_mlp_hidden']
                    if 'embed_dim_stim' in params and params['embed_dim_stim'] is not None:
                        cfg_eval.tbfm.module.embed_dim_stim = params['embed_dim_stim']
                    
                    OmegaConf.set_struct(cfg_eval, True)
                
                print(f"[GPU {gpu_id}] Config after applying params: latent_dim={cfg_eval.latent_dim}, "
                      f"num_bases={cfg_eval.tbfm.module.num_bases}, rr={cfg_eval.meta.basis_residual_rank}, "
                      f"mlp_hidden={cfg_eval.meta.residual_mlp_hidden}, embed_dim_stim={cfg_eval.tbfm.module.embed_dim_stim}")
                
                model_file = model_path / "model_nf_1.torch"
                if not model_file.exists():
                    model_file = model_path / "model.torch"
                
                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found for {model_key}")
                
                # Get embeddings for this model
                embeddings = embeddings_cache[model_key]
                
                # Verify config one more time right before building
                print(f"[GPU {gpu_id}] VERIFY before build_from_cfg: cfg_eval.latent_dim = {cfg_eval.latent_dim}")
                print(f"[GPU {gpu_id}] VERIFY type: {type(cfg_eval.latent_dim)}, value: {repr(cfg_eval.latent_dim)}")
                
                # Build model
                ms_eval = multisession.build_from_cfg(
                    cfg_eval,
                    data_train,
                    base_model_path=str(model_file),
                    device=device,
                )
                
                # Run TTA
                adapted_embeddings, strategy_results = multisession.test_time_adaptation(
                    cfg_eval,
                    ms_eval,
                    embeddings,
                    data_train,
                    epochs=tta_epochs,
                    data_test=data_test,
                    ae_warm_start=True,
                    adapt_ae=True,
                    support_size=support_size,
                    coadapt_embeddings=strategy_cfg["coadapt_embeddings"],
                    quiet=True,
                    progress_job_id=job_id,  # Pass job_id for progress updates
                )
                
                final_r2 = strategy_results["final_test_r2"]
                per_session_r2s = strategy_results.get("final_test_r2s", {})
                
                # Save adapted model
                if output_dir is not None:
                    adapted_model_dir = output_dir / "adapted_models" / f"{model_key}_support{support_size}_{strategy_key}"
                    adapted_model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save adapted model (TBFM only)
                    multisession.save_model(ms_eval, adapted_model_dir / "model_adapted.torch", tbfm_only=True)
                    
                    # Save adapted embeddings
                    torch.save(adapted_embeddings, adapted_model_dir / "embeddings_stim_adapted.torch")
                    
                    # Save metadata
                    metadata = {
                        "model_key": model_key,
                        "support_size": support_size,
                        "strategy_key": strategy_key,
                        "final_r2": final_r2,
                        "tta_epochs": tta_epochs,
                        "adapt_session_ids": adapt_session_ids,
                    }
                    torch.save(metadata, adapted_model_dir / "metadata.torch")
                
                # Clean up
                del ms_eval
                del strategy_results
                del adapted_embeddings
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Put result in queue
                result_queue.put({
                    "success": True,
                    "model": model_key,
                    "support_size": support_size,
                    "strategy": strategy_key,
                    "r2": final_r2,
                    "per_session_r2s": per_session_r2s,
                    "gpu_id": gpu_id,
                })
                
                print(f"[GPU {gpu_id}] Completed: Model={model_key}, Support={support_size}, Strategy={strategy_key}, R²={final_r2:.4f}")
                
                # Send notification for TTA completion
                try:
                    session_str = ",".join(adapt_session_ids[:3]) + ("..." if len(adapt_session_ids) > 3 else "")
                    
                    if job_id:
                        # Complete progress notification
                        notifications.complete_progress_notification(
                            job_id,
                            final_message=f"TTA complete!\nModel: {model_key}\nStrategy: {tta_strategies[strategy_key]['label']}\nSupport: {support_size}\nR²: {final_r2:.4f}\nSessions: {session_str}",
                            title="✓ TTA Complete"
                        )
                    else:
                        # Fallback to regular notification
                        notifications.notify_tta_complete(
                            model_name=model_key,
                            session=session_str,
                            support_size=support_size,
                            strategy=tta_strategies[strategy_key]["label"],
                            r2=final_r2,
                            output_dir=str(output_dir) if output_dir else None
                        )
                except Exception as notif_error:
                    print(f"[GPU {gpu_id}] Failed to send notification: {notif_error}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing job: {e}")
                result_queue.put({
                    "success": False,
                    "model": model_key,
                    "support_size": support_size,
                    "strategy": strategy_key,
                    "error": str(e),
                    "gpu_id": gpu_id,
                })
        
        except Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Worker error: {e}")
            break
    
    print(f"[GPU {gpu_id}] Worker finished")


def load_model_hyperparameters(model_path: Path) -> Dict:
    """
    Load hyperparameters from model directory.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary with model hyperparameters
    """
    hyperparam_file = model_path / "hyperparameters.torch"
    
    if hyperparam_file.exists():
        params = torch.load(hyperparam_file, map_location='cpu')
        print(f"Loaded hyperparameters from {hyperparam_file.name}")
        return params
    else:
        # Fallback: try to parse from directory name for backwards compatibility
        print(f"Warning: hyperparameters.torch not found in {model_path.name}, attempting to parse from name")
        return parse_model_hyperparameters(model_path)


def parse_model_hyperparameters(model_path: Path) -> Dict:
    """
    Parse hyperparameters from model directory name.
    
    Expected format: {num_bases}_{num_sessions}_rr{rr}_inner_ts{train_size}[_shuffle]_ld{latent_dim}_bs{batch_size}_mlp{mlp_hidden}_eds{embed_dim_stim}
    
    Returns:
        Dictionary with extracted hyperparameters
    """
    import re
    
    model_name = model_path.name
    params = {}
    
    # Extract residual rank
    match = re.search(r'_rr(\d+)', model_name)
    if match:
        params['basis_residual_rank'] = int(match.group(1))
    
    # Extract latent dimension
    match = re.search(r'_ld(\d+)', model_name)
    if match:
        params['latent_dim'] = int(match.group(1))
    
    # Extract MLP hidden dimension
    match = re.search(r'_mlp(\d+)', model_name)
    if match:
        params['residual_mlp_hidden'] = int(match.group(1))
    
    # Extract stim embedding dimension
    match = re.search(r'_eds(\d+)', model_name)
    if match:
        params['embed_dim_stim'] = int(match.group(1))
    
    # Extract num bases (at start of name)
    match = re.match(r'^(\d+)_', model_name)
    if match:
        params['num_bases'] = int(match.group(1))
    
    return params


def load_configuration(config_dir: Path, model_path: Path = None):
    """Load Hydra configuration and apply model-specific settings."""
    conf_dir = config_dir.resolve()

    if not conf_dir.exists():
        raise FileNotFoundError(f"Configuration directory not found: {conf_dir}")

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config")

    # Apply default model-specific configuration overrides
    cfg.training.epochs = 7001
    cfg.latent_dim = 85
    cfg.tbfm.module.num_bases = 100
    cfg.ae.training.lambda_ae_recon = 0.03
    cfg.ae.use_two_stage = False  # ts5000 models use single-stage LinearChannelAE
    cfg.ae.two_stage.freeze_only_shared = False
    cfg.ae.two_stage.lambda_mu = 0.01
    cfg.ae.two_stage.lambda_cov = 0.01
    cfg.tbfm.training.lambda_fro = 75.0
    cfg.meta.is_basis_residual = True
    cfg.meta.basis_residual_rank = 16
    cfg.meta.residual_mlp_hidden = 16
    cfg.tbfm.module.embed_dim_stim = 15
    cfg.meta.training.lambda_l2 = 1e-2
    cfg.meta.training.coadapt = False
    
    # Parse and apply model-specific hyperparameters if model_path provided
    if model_path is not None:
        params = load_model_hyperparameters(model_path)
        if params:
            print(f"Applying model-specific hyperparameters from {model_path.name}:")
            for key, value in params.items():
                if value is not None:  # Only print non-None values
                    print(f"  {key}: {value}")
            
            # Apply parsed parameters to config
            if 'latent_dim' in params:
                cfg.latent_dim = params['latent_dim']
            if 'num_bases' in params:
                cfg.tbfm.module.num_bases = params['num_bases']
            if 'basis_residual_rank' in params:
                cfg.meta.basis_residual_rank = params['basis_residual_rank']
            if 'residual_mlp_hidden' in params:
                cfg.meta.residual_mlp_hidden = params['residual_mlp_hidden']
            if 'embed_dim_stim' in params:
                cfg.tbfm.module.embed_dim_stim = params['embed_dim_stim']
    
    # Workaround: ae.py expects cfg.should_warm_start but config has it under cfg.ae.should_warm_start
    # Copy it to the global level if it exists
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg, 'ae') and hasattr(cfg.ae, 'should_warm_start'):
        cfg.should_warm_start = cfg.ae.should_warm_start
    else:
        cfg.should_warm_start = False
    
    # Workaround: ae.py expects cfg.should_warm_start but config has it under cfg.ae.should_warm_start
    # Copy it to the global level if it exists
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg, 'ae') and hasattr(cfg.ae, 'should_warm_start'):
        cfg.should_warm_start = cfg.ae.should_warm_start
    else:
        cfg.should_warm_start = True
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg, True)

    return cfg


def get_model_paths() -> Dict[str, Path]:
    """Return dictionary of available model paths."""
    return {
        # ts5000 models with different pretrain sizes (5, 12, 15, 20, 25)
        # "100_5_inner_ts5000": Path("test/100_5_rr16_inner_ts5000").resolve(),
        # "100_12_inner_ts5000": Path("test/100_12_rr16_inner_ts5000").resolve(),
        # "100_15_inner_ts5000": Path("test/100_15_rr16_inner_ts5000").resolve(),
        # "100_20_inner_ts5000": Path("test/100_20_rr16_inner_ts5000").resolve(),
        # "100_25_inner_ts5000": Path("test/100_25_rr16_inner_ts5000").resolve(),
        "100_25_inner_ts5000_shuffle": Path("test/100_25_rr16_inner_ts5000_shuffle").resolve(),
    }


def load_held_in_sessions(model_paths: Dict[str, Path]) -> Tuple[Dict, Dict]:
    """
    Load held-in and held-out session IDs for each model.

    Returns:
        Tuple of (held_in_sessions_map, held_out_sessions_map)
    """
    meta = dataset.load_meta(DATA_DIR)
    all_session_ids = [m.session_id for m in meta]

    held_in_sessions_map = {}
    held_out_sessions_map = {}

    for model_name, model_path in model_paths.items():
        # Find held-in sessions file
        hisi_path = None
        for candidate in ["hisi_nf_1.torch", "hisi.torch"]:
            path = model_path / candidate
            if path.exists():
                hisi_path = path
                break

        if hisi_path is None:
            raise FileNotFoundError(
                f"Missing held-in sessions file for {model_name} in {model_path}"
            )

        held_in_sessions = torch.load(hisi_path)
        held_in_sessions_map[model_name] = held_in_sessions

        held_out_sessions = sorted(set(all_session_ids) - set(held_in_sessions))
        held_out_sessions_map[model_name] = held_out_sessions

        print(
            f"{model_name}: {len(held_in_sessions)} held-in / "
            f"{len(held_out_sessions)} held-out sessions"
        )

    return held_in_sessions_map, held_out_sessions_map


def find_shared_held_out_sessions(held_out_sessions_map: Dict) -> List[str]:
    """Find sessions that are held-out across all models."""
    meta = dataset.load_meta(DATA_DIR)
    all_session_ids = [m.session_id for m in meta]

    shared_held_out = set(all_session_ids)
    for sessions in held_out_sessions_map.values():
        shared_held_out &= set(sessions)

    shared_held_out = sorted(shared_held_out)

    if not shared_held_out:
        raise ValueError("No shared held-out sessions found across all models")

    print(
        f"Shared held-out sessions ({len(shared_held_out)} total): "
        f"{shared_held_out[:5]}{'...' if len(shared_held_out) > 5 else ''}"
    )

    return shared_held_out


def select_adapt_sessions(
    shared_held_out_sessions: List[str],
    max_sessions: int,
    specific_session: str = None
) -> List[str]:
    """
    Select adaptation sessions from shared held-out sessions.

    Only includes sessions with cached rest embeddings.
    
    Args:
        shared_held_out_sessions: List of available held-out sessions
        max_sessions: Maximum number of sessions to select
        specific_session: If provided, use this specific session ID
    """
    if specific_session:
        # Verify the specific session exists and has cached embeddings
        if specific_session not in shared_held_out_sessions:
            raise ValueError(
                f"Specified session '{specific_session}' is not in shared held-out sessions.\n"
                f"Available sessions: {shared_held_out_sessions}"
            )
        
        path = os.path.join(DATA_DIR, specific_session, EMBEDDING_REST_SUBDIR, "er.torch")
        if not os.path.exists(path):
            raise ValueError(
                f"Specified session '{specific_session}' does not have cached rest embeddings at {path}"
            )
        
        adapt_session_ids = [specific_session]
        print(f"Using specified adaptation session: {specific_session}")
        return adapt_session_ids

    # Original logic for auto-selection
    adapt_session_ids = []

    for sid in shared_held_out_sessions:
        if len(adapt_session_ids) >= max_sessions:
            break

        path = os.path.join(DATA_DIR, sid, EMBEDDING_REST_SUBDIR, "er.torch")
        if os.path.exists(path):
            adapt_session_ids.append(sid)

    if not adapt_session_ids:
        raise ValueError(
            "No shared held-out sessions with cached rest embeddings found"
        )

    print(f"Selected {len(adapt_session_ids)} adaptation session(s): {adapt_session_ids}")
    return adapt_session_ids


def prepare_data(
    adapt_session_ids: List[str],
    window_size: int,
    batch_size_per_session: int,
    device: str
):
    """
    Load and prepare training/test data and rest embeddings.

    Returns:
        Tuple of (data_train, data_test, embeddings_rest)
    """
    batch_size = batch_size_per_session * len(adapt_session_ids)

    print(f"Loading data with batch size: {batch_size}")
    d, _ = multisession.load_stim_batched(
        batch_size=batch_size,
        window_size=window_size,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        held_in_session_ids=adapt_session_ids,
        num_held_out_sessions=0,
    )

    data_train, data_test = d.train_test_split(5000, test_cut=2500)

    # Load rest embeddings for adaptation sessions
    embeddings_rest = multisession.load_rest_embeddings(adapt_session_ids, device=device)

    print("Data prepared: train/test split complete")
    return data_train, data_test, embeddings_rest


def prepare_embeddings_for_model(
    model_key: str,
    held_in_sessions_map: Dict,
    adapt_session_ids: List[str],
    device: str
) -> Dict:
    """Load rest embeddings for a model's sessions plus adaptation sessions."""
    session_ids = held_in_sessions_map[model_key]
    embeds = multisession.load_rest_embeddings(session_ids, device=device)
    embeds.update(multisession.load_rest_embeddings(adapt_session_ids, device=device))
    return embeds


def clone_cfg_for_eval(base_cfg, vanilla = False):
    """Deep copy Hydra config to prevent mutations from leaking across runs."""
    cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(base_cfg, resolve=True)))
    def cfg_identity(cfg, dim):
        cfg.ae.training.coadapt = False
        cfg.ae.warm_start_is_identity = True
        cfg.latent_dim = dim
        
    def cfg_base(cfg, dim):
        cfg_identity(cfg, dim)
        # cfg.training.grad_clip = 2.0
        # cfg.tbfm.training.lambda_ortho = 0.05
        # cfg.tbfm.module.use_film_bases = False
        cfg.tbfm.module.num_bases = 12
        cfg.tbfm.module.latent_dim = 2
        cfg.training.epochs = 12001
    if vanilla:
        cfg_base(cfg, cfg.latent_dim)
    return cfg

def get_tta_strategies() -> Dict:
    """Return available TTA strategies."""
    return {
        "coadapt": {
            "label": "Co-Adaptation",
            "coadapt_embeddings": True
        },
        "maml": {
            "label": "MAML (Meta-Learning)",
            "coadapt_embeddings": False
        },
    }


def train_vanilla_tbfm(
    cfg,
    data_train,
    data_test,
    support_size: int,
    epochs: int,
    device: str,
    quiet: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Train a vanilla (single-session) TBFM model on the support set.

    Uses direct TBFM class instantiation like in the demo, not the multisession infrastructure.

    Args:
        cfg: Config object
        data_train: Training data for adaptation
        data_test: Test data for evaluation
        support_size: Number of samples to use for training
        epochs: Number of training epochs
        device: Device to train on
        quiet: If True, suppress progress messages

    Returns:
        Tuple of (final_test_r2, per_session_r2s_dict)
    """
    from torcheval.metrics.functional import r2_score
    from tbfm import tbfm as tbfm_module

    if not quiet:
        print("Training vanilla TBFM on support set...")

    device_obj = torch.device(device) if isinstance(device, str) else device

    # Materialize support set
    _, data_support = utils.iter_loader(iter(data_train), data_train, device=str(device_obj))

    # Filter support data to requested size
    data_support, _ = multisession.split_support_query_sessions(data_support, support_size)

    # Since sessions may have different dimensions, train a separate model for each
    # and average the results
    session_r2s = []
    
    for session_id, (runway_train, stiminds_train, y_train) in data_support.items():
        if not quiet:
            print(f"  Training vanilla TBFM for session: {session_id}")
        
        # Get hyperparameters from config
        cfg_eval = clone_cfg_for_eval(cfg, vanilla=True)
        in_dim = runway_train.shape[-1]
        runway_len = runway_train.shape[1]
        trial_len = y_train.shape[1]
        
        # The new Bases class expects unpacked stiminds (batch, stimdim)
        # stiminds_train is already in the correct format: (batch, 2)
        # However, Bases.__init__ does `in_dim = stimdim - 1` (line 79), expecting
        # stimdim to include a clock dimension. So we need to tell TBFM that stimdim=3
        # even though we only pass 2 features, to account for this subtraction.
        stim_dim = stiminds_train.shape[-1] + 1  # +1 to account for clock dimension subtraction
        
        num_bases = cfg_eval.tbfm.module.num_bases
        latent_dim = cfg_eval.latent_dim
        basis_depth = cfg_eval.tbfm.module.basis_depth
        lambda_fro = cfg_eval.tbfm.training.lambda_fro
        test_interval = cfg_eval.training.test_interval

        # Create vanilla TBFM model directly (single-session, no meta-learning)
        _tbfm = tbfm_module.TBFM(
            in_dim=in_dim,
            stimdim=stim_dim,
            runway=runway_len,
            num_bases=num_bases,
            trial_len=trial_len,
            batchy=y_train,  # Use support set for normalization
            latent_dim=latent_dim,
            basis_depth=basis_depth,
            zscore=True,
            use_meta_learning=False,  # Vanilla TBFM doesn't use meta-learning
            device=device_obj,
        )

        # Normalize y values using model's normalizer
        y_train_norm = _tbfm.normalize(y_train)

        # Get optimizer
        optim = _tbfm.get_optim(lr=cfg_eval.tbfm.training.optim.lr_head)

        # Training loop
        _tbfm.train()

        for epoch in range(epochs):
            # Training step
            optim.zero_grad()

            # Forward pass on support set
            y_pred = _tbfm(runway_train, stiminds_train)

            # Compute loss
            loss = torch.nn.functional.mse_loss(y_pred, y_train_norm)

            # Add regularization
            if lambda_fro > 0:
                loss_reg = _tbfm.get_weighting_reg()
                loss = loss + lambda_fro * loss_reg

            loss.backward()
            optim.step()

        # Final evaluation on test set for this session
        _tbfm.eval()
        with torch.no_grad():
            test_r2_acc = 0.0
            test_batch_count = 0

            for test_batch_dict in data_test:
                # Only evaluate on the matching session
                if session_id not in test_batch_dict:
                    continue
                    
                runway_test, stiminds_test, y_test = test_batch_dict[session_id]
                runway_test = runway_test.to(device_obj)
                stiminds_test = stiminds_test.to(device_obj)
                y_test = y_test.to(device_obj)

                # Normalize y_test using the model's normalizer
                y_test_norm = _tbfm.normalize(y_test)

                y_pred_test = _tbfm(runway_test, stiminds_test)
                test_r2 = r2_score(y_pred_test.flatten(), y_test_norm.flatten())

                test_r2_acc += test_r2.item()
                test_batch_count += 1

            if test_batch_count > 0:
                session_final_r2 = test_r2_acc / test_batch_count
                session_r2s.append(session_final_r2)
                
                if not quiet:
                    print(f"    Session {session_id} final test R²={session_final_r2:.4f}")
        
        # Clean up this session's model
        del _tbfm
        del optim
        torch.cuda.empty_cache()
    
    # Average R² across all sessions
    final_test_r2 = sum(session_r2s) / len(session_r2s) if session_r2s else 0.0
    
    # Create per-session R² dict (using session IDs from data_support)
    per_session_r2s = {session_id: r2 for session_id, r2 in zip(data_support.keys(), session_r2s)}

    if not quiet:
        print(f"Vanilla TBFM training complete. Average test R² across {len(session_r2s)} sessions: {final_test_r2:.4f}")

    return final_test_r2, per_session_r2s


def train_fresh_tbfm_no_multisession(
    cfg,
    data_train,
    data_test,
    support_size: int,
    epochs: int,
    device: str,
    quiet: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Train fresh TBFM models from scratch per session (no multisession features).

    Trains a separate model for each session using the multisession infrastructure
    but with all multisession features disabled (no shared parameters, no meta-learning).
    Returns the average R² across all sessions.

    Args:
        cfg: Config object
        data_train: Training data for adaptation
        data_test: Test data for evaluation
        support_size: Number of samples to use for training
        epochs: Number of training epochs
        device: Device to train on
        quiet: If True, suppress progress messages

    Returns:
        Tuple of (final_test_r2, per_session_r2s_dict)
    """
    from torcheval.metrics.functional import r2_score

    if not quiet:
        print("Training fresh TBFM (per-session, no multisession) on support set...")

    device_obj = torch.device(device) if isinstance(device, str) else device

    # Materialize support set
    _, data_support = utils.iter_loader(iter(data_train), data_train, device=str(device_obj))

    # Filter support data to requested size
    data_support, _ = multisession.split_support_query_sessions(data_support, support_size)

    # Train a separate model for each session (like vanilla TBFM)
    # This handles heterogeneous sessions naturally
    session_r2s = []
    
    for session_id, (runway_train, stiminds_train, y_train) in data_support.items():
        if not quiet:
            print(f"  Training fresh TBFM for session: {session_id}")
        
        # Use the same approach as vanilla TBFM but with multisession infrastructure
        from tbfm import tbfm as tbfm_module
        
        # Clone config and disable multisession features for this session
        cfg_eval = clone_cfg_for_eval(cfg, vanilla=True)
        
        
        # Get dimensions for this session
        y_dim = y_train.shape[-1]
        runway_len = runway_train.shape[1]
        trial_len = y_train.shape[1]
        stim_dim = stiminds_train.shape[-1] + 1  # +1 to account for clock dimension
        
        num_bases = cfg_eval.tbfm.module.num_bases
        latent_dim = y_dim  # Use full dimension for identity mapping
        basis_depth = cfg_eval.tbfm.module.basis_depth
        lambda_fro = cfg_eval.tbfm.training.lambda_fro
        
        # Create TBFM with identity AE (effectively just TBFM on raw data)
        # We'll use the normalizer but skip the actual AE
        _tbfm = tbfm_module.TBFM(
            in_dim=y_dim,
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
        optim = _tbfm.get_optim(lr=cfg_eval.tbfm.training.optim.lr_head)
        
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
        
        # Evaluate on test set for this session
        _tbfm.eval()
        with torch.no_grad():
            test_r2_acc = 0.0
            test_batch_count = 0
            
            for test_batch_dict in data_test:
                if session_id not in test_batch_dict:
                    continue
                
                runway_test, stiminds_test, y_test = test_batch_dict[session_id]
                runway_test = runway_test.to(device_obj)
                stiminds_test = stiminds_test.to(device_obj)
                y_test = y_test.to(device_obj)
                
                y_test_norm = _tbfm.normalize(y_test)
                y_pred_test = _tbfm(runway_test, stiminds_test)
                
                from torcheval.metrics.functional import r2_score
                test_r2 = r2_score(y_pred_test.flatten(), y_test_norm.flatten())
                test_r2_acc += test_r2.item()
                test_batch_count += 1
            
            if test_batch_count > 0:
                session_final_r2 = test_r2_acc / test_batch_count
                session_r2s.append(session_final_r2)
                
                if not quiet:
                    print(f"    Session {session_id} final test R²={session_final_r2:.4f}")
        
        # Clean up
        del _tbfm
        del optim
        torch.cuda.empty_cache()
    
    # Average R² across all sessions
    final_test_r2 = sum(session_r2s) / len(session_r2s) if session_r2s else 0.0
    
    # Create per-session R² dict (using session IDs from data_support)
    per_session_r2s = {session_id: r2 for session_id, r2 in zip(data_support.keys(), session_r2s)}

    if not quiet:
        print(f"Fresh TBFM training complete. Average test R² across {len(session_r2s)} sessions: {final_test_r2:.4f}")

    return final_test_r2, per_session_r2s


def run_tta_sweep_multi_gpu(
    model_paths: Dict[str, Path],
    held_in_sessions_map: Dict,
    support_sizes: List[int],
    adapt_session_ids: List[str],
    cfg,
    data_train,
    data_test,
    tta_epochs: int,
    device: str,
    gpu_ids: List[int],
    output_dir: Path,
    include_vanilla_tbfm: bool,
    vanilla_tbfm_epochs: int,
    include_fresh_tbfm: bool,
    fresh_tbfm_epochs: int,
    jobs: List,
    tta_strategies: Dict,
    embeddings_cache: Dict,
    timestamp: str,
    log_path: Path,
) -> Dict:
    """
    Multi-GPU version of run_tta_sweep using multiprocessing.
    
    Distributes TTA jobs across multiple GPUs in parallel.
    """
    model_names = list(model_paths.keys())
    
    # Storage for results
    tta_comparison = {
        model_key: {strategy_key: [] for strategy_key in tta_strategies}
        for model_key in model_names
    }
    vanilla_tbfm_results = {}
    fresh_tbfm_results = {}
    tta_runs = []
    
    # Separate TTA jobs from baseline jobs
    tta_jobs = [j for j in jobs if len(j) == 3]
    baseline_jobs = [j for j in jobs if len(j) == 2]
    
    # Process baselines on single GPU (typically fast and not worth parallelizing)
    if baseline_jobs:
        print(f"\\nProcessing {len(baseline_jobs)} baseline jobs on single GPU...")
        device = f"cuda:{gpu_ids[0]}"
        
        for job in tqdm.tqdm(baseline_jobs, desc="Baseline jobs"):
            model_key, support_size = job
            
            if model_key == "vanilla_tbfm":
                final_r2 = train_vanilla_tbfm(
                    cfg,
                    data_train,
                    data_test,
                    support_size,
                    vanilla_tbfm_epochs,
                    device,
                    quiet=True,
                )
                vanilla_tbfm_results[support_size] = final_r2
                tta_runs.append({
                    "model": "vanilla_tbfm",
                    "support_size": support_size,
                    "strategy": "vanilla_tbfm",
                    "r2": final_r2,
                })
            elif model_key == "fresh_tbfm":
                final_r2 = train_fresh_tbfm_no_multisession(
                    cfg,
                    data_train,
                    data_test,
                    support_size,
                    fresh_tbfm_epochs,
                    device,
                    quiet=True,
                )
                fresh_tbfm_results[support_size] = final_r2
                tta_runs.append({
                    "model": "fresh_tbfm",
                    "support_size": support_size,
                    "strategy": "fresh_tbfm",
                    "r2": final_r2,
                })
    
    # Multi-GPU processing for TTA jobs
    if tta_jobs:
        print(f"\nProcessing {len(tta_jobs)} TTA jobs across {len(gpu_ids)} GPUs...")
        
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        mp_context = mp.get_context('spawn')
        
        # Get configuration parameters for workers
        window_size = cfg.data.trial_len
        batch_size_per_session = 7500  # You may want to pass this as a parameter
        
        # Convert config to dict for serialization
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Convert model paths to strings
        model_paths_dict = {k: str(v) for k, v in model_paths.items()}
        
        # Create job and result queues using spawn context
        job_queue = mp_context.Queue()
        result_queue = mp_context.Queue()
        
        # Populate job queue
        for job in tta_jobs:
            job_queue.put(job)
        
        # Add poison pills (one per worker)
        for _ in gpu_ids:
            job_queue.put(None)
        
        # Start worker processes using spawn context
        workers = []
        for gpu_id in gpu_ids:
            p = mp_context.Process(
                target=gpu_worker,
                args=(
                    gpu_id,
                    job_queue,
                    result_queue,
                    cfg_dict,
                    model_paths_dict,
                    adapt_session_ids,
                    window_size,
                    batch_size_per_session,
                    tta_epochs,
                    tta_strategies,
                    output_dir,
                ),
            )
            p.start()
            workers.append(p)
        
        # Collect results with progress bar
        completed = 0
        progress = tqdm.tqdm(total=len(tta_jobs), desc="TTA jobs")
        
        with open(log_path, "w", buffering=1, encoding="utf-8") as log_stream:
            def log_line(message=""):
                print(message, flush=True)
                log_stream.write(message + "\\n")
                log_stream.flush()
            
            log_line(f"TTA sweep started at {timestamp}")
            log_line(f"Multi-GPU mode: Using GPUs {gpu_ids}")
            log_line(f"Adapt session(s): {adapt_session_ids}")
            log_line(f"Support sizes: {support_sizes}")
            log_line(f"TTA epochs: {tta_epochs}")
            log_line(f"Total TTA jobs: {len(tta_jobs)}")
            log_line("")
            
            while completed < len(tta_jobs):
                try:
                    result = result_queue.get(timeout=1)
                    
                    if result["success"]:
                        model_key = result["model"]
                        support_size = result["support_size"]
                        strategy_key = result["strategy"]
                        final_r2 = result["r2"]
                        per_session_r2s = result.get("per_session_r2s", {})
                        gpu_id = result["gpu_id"]
                        
                        tta_comparison[model_key][strategy_key].append((support_size, final_r2))
                        tta_runs.append({
                            "model": model_key,
                            "support_size": support_size,
                            "strategy": strategy_key,
                            "r2": final_r2,
                            "per_session_r2s": per_session_r2s,
                        })
                        
                        strategy_label = tta_strategies[strategy_key]["label"]
                        log_line(
                            f"[GPU {gpu_id}] Complete | Strategy={strategy_label:<24} | "
                            f"Support={support_size:>5} | Model={model_key:<20} | R²={final_r2:.4f}"
                        )
                    else:
                        log_line(f"[GPU {result['gpu_id']}] FAILED: {result['model']}, {result['support_size']}, {result['strategy']}: {result['error']}")
                    
                    completed += 1
                    progress.update(1)
                    
                except Empty:
                    continue
            
            progress.close()
            
            # Wait for all workers to finish
            for p in workers:
                p.join()
            
            log_line("\\n" + "=" * 80)
            log_line("TTA Sweep Complete - Summary")
            log_line("=" * 80)
            
            # Print results
            for entry in sorted(tta_runs, key=lambda x: (x.get("strategy", ""), x["support_size"], x["model"])):
                if entry["strategy"] not in ["vanilla_tbfm", "fresh_tbfm"]:
                    strategy_label = tta_strategies[entry["strategy"]]["label"]
                    log_line(
                        f"Strategy={strategy_label:<24} | Support={entry['support_size']:>5} | "
                        f"Model={entry['model']:<20} | R²={entry['r2']:.4f}"
                    )
            
            if vanilla_tbfm_results:
                log_line("\\nVanilla TBFM Baseline:")
                for entry in sorted([e for e in tta_runs if e.get("strategy") == "vanilla_tbfm"], key=lambda x: x["support_size"]):
                    log_line(f"  Support={entry['support_size']:>5} | R²={entry['r2']:.4f}")
            
            if fresh_tbfm_results:
                log_line("\\nFresh TBFM (no multisession) Baseline:")
                for entry in sorted([e for e in tta_runs if e.get("strategy") == "fresh_tbfm"], key=lambda x: x["support_size"]):
                    log_line(f"  Support={entry['support_size']:>5} | R²={entry['r2']:.4f}")
    
    return {
        "metadata": {
            "timestamp": timestamp,
            "adapt_session_ids": adapt_session_ids,
            "support_sizes": support_sizes,
            "models": model_names,
            "tta_epochs": tta_epochs,
            "include_vanilla_tbfm": include_vanilla_tbfm,
            "include_fresh_tbfm": include_fresh_tbfm,
            "multi_gpu": True,
            "gpu_ids": gpu_ids,
        },
        "strategies": tta_strategies,
        "grid": tta_comparison,
        "vanilla_tbfm_results": vanilla_tbfm_results,
        "fresh_tbfm_results": fresh_tbfm_results,
        "runs": tta_runs,
    }


def run_tta_sweep(
    model_paths: Dict[str, Path],
    held_in_sessions_map: Dict,
    support_sizes: List[int],
    adapt_session_ids: List[str],
    cfg,
    data_train,
    data_test,
    tta_epochs: int,
    device: str,
    output_dir: Path,
    include_vanilla_tbfm: bool = False,
    vanilla_tbfm_epochs: int = 7001,
    include_fresh_tbfm: bool = False,
    fresh_tbfm_epochs: int = 7001,
    use_multi_gpu: bool = False,
    gpu_ids: List[int] = None,
) -> Dict:
    """
    Run comprehensive TTA sweep across models, support sizes, and strategies.

    Args:
        model_paths: Dictionary mapping model names to paths
        held_in_sessions_map: Dictionary of held-in sessions per model
        support_sizes: List of support set sizes to evaluate
        adapt_session_ids: Session IDs to use for adaptation
        cfg: Hydra configuration object
        data_train: Training data
        data_test: Test data
        tta_epochs: Number of epochs for TTA
        device: CUDA device to use (ignored if use_multi_gpu is True)
        output_dir: Output directory for results
        include_vanilla_tbfm: If True, include vanilla TBFM baseline
        vanilla_tbfm_epochs: Number of epochs for vanilla TBFM training
        include_fresh_tbfm: If True, include fresh TBFM (no multisession) baseline
        fresh_tbfm_epochs: Number of epochs for fresh TBFM training
        use_multi_gpu: If True, distribute jobs across multiple GPUs
        gpu_ids: List of GPU IDs to use (if None, uses all available)

    Returns:
        Dictionary containing all results and metadata
    """
    tta_strategies = get_tta_strategies()
    model_names = list(model_paths.keys())

    # Cache embeddings per model
    print("\nCaching embeddings for all models...")
    embeddings_cache = {
        name: prepare_embeddings_for_model(
            name, held_in_sessions_map, adapt_session_ids, device
        )
        for name in model_names
    }

    # Storage for results
    tta_comparison = {
        model_key: {strategy_key: [] for strategy_key in tta_strategies}
        for model_key in model_names
    }
    vanilla_tbfm_results = {}  # Dict to store vanilla TBFM results
    fresh_tbfm_results = {}  # Dict to store fresh TBFM results
    tta_runs = []

    # Create job grid
    jobs = [
        (model_key, support_size, strategy_key)
        for model_key in model_names
        for support_size in support_sizes
        for strategy_key in tta_strategies.keys()
    ]

    # Add vanilla TBFM jobs if requested
    if include_vanilla_tbfm:
        for support_size in support_sizes:
            jobs.append(("vanilla_tbfm", support_size))

    # Add fresh TBFM jobs if requested
    if include_fresh_tbfm:
        for support_size in support_sizes:
            jobs.append(("fresh_tbfm", support_size))

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"tta_sweep_{timestamp}.log"

    print(f"\n{'=' * 80}")
    print("Running TTA Sweep")
    print(f"{'=' * 80}")
    print(f"Models: {model_names}")
    print(f"Support sizes: {support_sizes}")
    print(f"Strategies: {list(tta_strategies.keys())}")
    baselines = []
    if include_vanilla_tbfm:
        baselines.append("Vanilla TBFM")
    if include_fresh_tbfm:
        baselines.append("Fresh TBFM (no multisession)")
    if baselines:
        print(f"Baselines: {', '.join(baselines)}")
    print(f"Total jobs: {len(jobs)}")
    
    if use_multi_gpu:
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        print(f"Multi-GPU mode: Using GPUs {gpu_ids}")
    else:
        print(f"Single-GPU mode: Using device {device}")
    
    print(f"Logging to: {log_path}")
    print(f"{'=' * 80}\n")
    
    # Multi-GPU execution path
    if use_multi_gpu:
        return run_tta_sweep_multi_gpu(
            model_paths,
            held_in_sessions_map,
            support_sizes,
            adapt_session_ids,
            cfg,
            data_train,
            data_test,
            tta_epochs,
            device,
            gpu_ids,
            output_dir,
            include_vanilla_tbfm,
            vanilla_tbfm_epochs,
            include_fresh_tbfm,
            fresh_tbfm_epochs,
            jobs,
            tta_strategies,
            embeddings_cache,
            timestamp,
            log_path,
        )

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_stream:
        def log_line(message=""):
            print(message, flush=True)
            log_stream.write(message + "\n")
            log_stream.flush()

        log_line(f"TTA sweep started at {timestamp}")
        log_line(f"Adapt session(s): {adapt_session_ids}")
        log_line(f"Support sizes: {support_sizes}")
        log_line(f"TTA epochs: {tta_epochs}")
        log_line("")

        progress = tqdm.tqdm(jobs, desc="TTA grid", leave=False)
        for job in progress:
            # Handle TTA jobs (3-tuple) and baseline jobs (2-tuple)
            if len(job) == 3:
                model_key, support_size, strategy_key = job
                is_baseline = False
            else:
                model_key, support_size = job
                strategy_key = None
                is_baseline = True

            if is_baseline:
                # Determine which baseline this is
                if model_key == "vanilla_tbfm":
                    baseline_name = "Vanilla TBFM"
                    status_msg = (
                        f"Running | Strategy={baseline_name:<24} | "
                        f"Support={support_size:>5} | Model={'N/A':<20}"
                    )
                    log_line(status_msg)
                    progress.set_postfix({
                        "model": "vanilla_tbfm",
                        "support": support_size,
                        "strategy": "baseline",
                    })

                    # Train vanilla TBFM
                    final_r2 = train_vanilla_tbfm(
                        cfg,
                        data_train,
                        data_test,
                        support_size,
                        vanilla_tbfm_epochs,
                        device,
                        quiet=True,
                    )

                    vanilla_tbfm_results[support_size] = final_r2
                    tta_runs.append({
                        "model": "vanilla_tbfm",
                        "support_size": support_size,
                        "strategy": "vanilla_tbfm",
                        "r2": final_r2,
                    })

                    log_line(
                        f"Complete  | Strategy={baseline_name:<24} | "
                        f"Support={support_size:>5} | Model={'N/A':<20} | R²={final_r2:.4f}"
                    )

                elif model_key == "fresh_tbfm":
                    baseline_name = "Fresh TBFM (no MS)"
                    status_msg = (
                        f"Running | Strategy={baseline_name:<24} | "
                        f"Support={support_size:>5} | Model={'N/A':<20}"
                    )
                    log_line(status_msg)
                    progress.set_postfix({
                        "model": "fresh_tbfm",
                        "support": support_size,
                        "strategy": "baseline",
                    })

                    # Train fresh TBFM (no multisession)
                    final_r2 = train_fresh_tbfm_no_multisession(
                        cfg,
                        data_train,
                        data_test,
                        support_size,
                        fresh_tbfm_epochs,
                        device,
                        quiet=True,
                    )

                    fresh_tbfm_results[support_size] = final_r2
                    tta_runs.append({
                        "model": "fresh_tbfm",
                        "support_size": support_size,
                        "strategy": "fresh_tbfm",
                        "r2": final_r2,
                    })

                    log_line(
                        f"Complete  | Strategy={baseline_name:<24} | "
                        f"Support={support_size:>5} | Model={'N/A':<20} | R²={final_r2:.4f}"
                    )

            else:
                strategy_cfg = tta_strategies[strategy_key]

                status_msg = (
                    f"Running | Strategy={strategy_cfg['label']:<24} | "
                    f"Support={support_size:>5} | Model={model_key:<20}"
                )
                log_line(status_msg)
                progress.set_postfix({
                    "model": model_key,
                    "support": support_size,
                    "strategy": strategy_key,
                })

                # Clone config and apply model-specific hyperparameters
                cfg_eval = clone_cfg_for_eval(cfg)
                
                # Parse and apply model-specific hyperparameters from path
                model_path = model_paths[model_key]
                params = load_model_hyperparameters(model_path)
                if params:
                    if 'latent_dim' in params and params['latent_dim'] is not None:
                        cfg_eval.latent_dim = params['latent_dim']
                    if 'num_bases' in params and params['num_bases'] is not None:
                        cfg_eval.tbfm.module.num_bases = params['num_bases']
                    if 'basis_residual_rank' in params and params['basis_residual_rank'] is not None:
                        cfg_eval.meta.basis_residual_rank = params['basis_residual_rank']
                    if 'residual_mlp_hidden' in params and params['residual_mlp_hidden'] is not None:
                        cfg_eval.meta.residual_mlp_hidden = params['residual_mlp_hidden']
                    if 'embed_dim_stim' in params and params['embed_dim_stim'] is not None:
                        cfg_eval.tbfm.module.embed_dim_stim = params['embed_dim_stim']

                # Find model file
                model_file = model_paths[model_key] / "model_nf_1.torch"
                if not model_file.exists():
                    model_file = model_paths[model_key] / "model.torch"

                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found for {model_key}")

                # Build model
                ms_eval = multisession.build_from_cfg(
                    cfg_eval,
                    data_train,
                    base_model_path=str(model_file),
                    device=device,
                )

                # Run TTA
                adapted_embeddings, strategy_results = multisession.test_time_adaptation(
                    cfg_eval,
                    ms_eval,
                    embeddings_cache[model_key],
                    data_train,
                    epochs=tta_epochs,
                    data_test=data_test,
                    ae_warm_start=True,
                    adapt_ae=True,
                    support_size=support_size,
                    coadapt_embeddings=strategy_cfg["coadapt_embeddings"],
                    quiet=True,
                )

                # Store results
                final_r2 = strategy_results["final_test_r2"]
                tta_comparison[model_key][strategy_key].append((support_size, final_r2))
                per_session_r2s = strategy_results.get("final_test_r2s", {})
                tta_runs.append({
                    "model": model_key,
                    "support_size": support_size,
                    "strategy": strategy_key,
                    "r2": final_r2,
                    "per_session_r2s": per_session_r2s,
                })

                # Save adapted model
                adapted_model_dir = output_dir / "adapted_models" / f"{model_key}_support{support_size}_{strategy_key}"
                adapted_model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save adapted model (TBFM only)
                multisession.save_model(ms_eval, adapted_model_dir / "model_adapted.torch", tbfm_only=True)
                
                # Save adapted embeddings
                torch.save(adapted_embeddings, adapted_model_dir / "embeddings_stim_adapted.torch")
                
                # Save metadata
                metadata = {
                    "model_key": model_key,
                    "support_size": support_size,
                    "strategy_key": strategy_key,
                    "final_r2": final_r2,
                    "tta_epochs": tta_epochs,
                    "adapt_session_ids": adapt_session_ids,
                }
                torch.save(metadata, adapted_model_dir / "metadata.torch")

                log_line(
                    f"Complete  | Strategy={strategy_cfg['label']:<24} | "
                    f"Support={support_size:>5} | Model={model_key:<20} | R²={final_r2:.4f}"
                )

                # Release GPU memory immediately after each strategy
                del ms_eval
                del strategy_results
                del adapted_embeddings
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.synchronize()  # Ensure GPU operations complete

        log_line("\n" + "=" * 80)
        log_line("TTA Sweep Complete - Summary")
        log_line("=" * 80)
        
        # Print TTA results
        for entry in sorted(tta_runs, key=lambda x: (x["strategy"], x["support_size"], x["model"])):
            if entry["strategy"] not in ["vanilla_tbfm", "fresh_tbfm"]:
                strategy_label = tta_strategies[entry["strategy"]]["label"]
                log_line(
                    f"Strategy={strategy_label:<24} | Support={entry['support_size']:>5} | "
                    f"Model={entry['model']:<20} | R²={entry['r2']:.4f}"
                )
        
        # Print vanilla TBFM results if included
        if include_vanilla_tbfm:
            log_line("\nVanilla TBFM Baseline:")
            for entry in sorted([e for e in tta_runs if e["strategy"] == "vanilla_tbfm"], key=lambda x: x["support_size"]):
                log_line(
                    f"Support={entry['support_size']:>5} | R²={entry['r2']:.4f}"
                )

        # Print fresh TBFM results if included
        if include_fresh_tbfm:
            log_line("\nFresh TBFM (no multisession) Baseline:")
            for entry in sorted([e for e in tta_runs if e["strategy"] == "fresh_tbfm"], key=lambda x: x["support_size"]):
                log_line(
                    f"Support={entry['support_size']:>5} | R²={entry['r2']:.4f}"
                )

    return {
        "metadata": {
            "timestamp": timestamp,
            "adapt_session_ids": adapt_session_ids,
            "support_sizes": support_sizes,
            "models": model_names,
            "tta_epochs": tta_epochs,
            "include_vanilla_tbfm": include_vanilla_tbfm,
            "include_fresh_tbfm": include_fresh_tbfm,
        },
        "strategies": tta_strategies,
        "grid": tta_comparison,
        "vanilla_tbfm_results": vanilla_tbfm_results,
        "fresh_tbfm_results": fresh_tbfm_results,
        "runs": tta_runs,
    }


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON file and per-session R² scores to CSV."""
    timestamp = results["metadata"]["timestamp"]
    json_path = output_dir / f"tta_support_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {json_path}")
    
    # Save per-session R² scores to CSV
    csv_path = output_dir / f"tta_support_{timestamp}_per_session.csv"
    save_per_session_csv(results, csv_path)
    
    return json_path


def save_per_session_csv(results: Dict, csv_path: Path):
    """Save per-session R² scores to CSV file."""
    rows = []
    
    for run in results["runs"]:
        model = run["model"]
        support_size = run["support_size"]
        strategy = run["strategy"]
        overall_r2 = run["r2"]
        per_session_r2s = run.get("per_session_r2s", {})
        
        if per_session_r2s:
            # Create a row for each session
            for session_id, session_r2 in per_session_r2s.items():
                rows.append({
                    "model": model,
                    "strategy": strategy,
                    "support_size": support_size,
                    "session_id": session_id,
                    "session_r2": session_r2,
                    "overall_r2": overall_r2,
                })
        else:
            # No per-session data, just record the overall
            rows.append({
                "model": model,
                "strategy": strategy,
                "support_size": support_size,
                "session_id": "(overall)",
                "session_r2": overall_r2,
                "overall_r2": overall_r2,
            })
    
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["model", "strategy", "support_size", "session_id", "session_r2", "overall_r2"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Per-session R² scores saved to: {csv_path}")


def plot_results(results: Dict, output_dir: Path, show: bool = False):
    """
    Create and save comparison plots.

    Args:
        results: Results dictionary from run_tta_sweep
        output_dir: Directory to save plot
        show: Whether to display plot interactively
    """
    timestamp = results["metadata"]["timestamp"]
    plot_path = output_dir / f"tta_support_{timestamp}.png"

    tta_strategies = results["strategies"]
    tta_comparison = results["grid"]
    vanilla_tbfm_results = results.get("vanilla_tbfm_results", {})
    fresh_tbfm_results = results.get("fresh_tbfm_results", {})
    support_sizes = results["metadata"]["support_sizes"]
    model_names = results["metadata"]["models"]
    include_vanilla_tbfm = results["metadata"].get("include_vanilla_tbfm", False)
    include_fresh_tbfm = results["metadata"].get("include_fresh_tbfm", False)

    fig, axes = plt.subplots(1, len(tta_strategies), figsize=(16, 6), sharey=True)
    if len(tta_strategies) == 1:
        axes = [axes]

    for ax, (strategy_key, strategy_cfg) in zip(axes, tta_strategies.items()):
        for model_key in model_names:
            data_points = sorted(tta_comparison[model_key][strategy_key], key=lambda x: x[0])
            support_vals = [dp[0] for dp in data_points]
            r2_vals = [dp[1] for dp in data_points]

            ax.plot(
                support_vals,
                r2_vals,
                marker="o",
                linewidth=2,
                label=model_key,
            )

        # Add vanilla TBFM baseline if available
        if include_vanilla_tbfm and vanilla_tbfm_results:
            support_vals = sorted(vanilla_tbfm_results.keys())
            r2_vals = [vanilla_tbfm_results[s] for s in support_vals]

            ax.plot(
                support_vals,
                r2_vals,
                marker="s",
                linewidth=2.5,
                linestyle="--",
                label="Vanilla TBFM",
                color="black",
            )

        # Add fresh TBFM baseline if available
        if include_fresh_tbfm and fresh_tbfm_results:
            support_vals = sorted(fresh_tbfm_results.keys())
            r2_vals = [fresh_tbfm_results[s] for s in support_vals]

            ax.plot(
                support_vals,
                r2_vals,
                marker="^",
                linewidth=2.5,
                linestyle=":",
                label="Fresh TBFM (no MS)",
                color="red",
            )

        ax.set_title(strategy_cfg["label"])
        ax.set_xlabel("Support Set Size")
        ax.set_xscale("log")
        ax.set_xticks(support_sizes)
        ax.set_xticklabels([str(s) for s in support_sizes], rotation=45)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test R²")
    axes[-1].legend(title="Model", bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.suptitle("TTA Performance vs Support Size")
    plt.tight_layout()

    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return plot_path


def main():
    """Main entry point for TTA evaluation script."""
    args = parse_args()
    
    try:
        _main_impl(args)
    except Exception as e:
        notifications.notify_error(
            script_name="tta_testing.py",
            error=e,
            context=f"TTA evaluation"
        )
        raise


def _main_impl(args):
    """Implementation of main function (separated for error handling)."""

    # Setup
    if not args.use_multi_gpu:
        setup_environment(args.cuda_device)
    device = "cuda"

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = load_configuration(args.config_dir)
    
    # Override inner_steps from command line
    cfg.meta.training.inner_steps = args.tta_inner_steps
    
    # Configure progressive unfreezing parameters
    cfg.meta.training.progressive_unfreezing_threshold = args.progressive_unfreezing_threshold
    cfg.meta.training.unfreeze_basis_weights = args.unfreeze_basis_weights
    cfg.meta.training.unfreeze_bases = args.unfreeze_bases
    cfg.meta.training.basis_weight_lr = args.basis_weight_lr
    cfg.meta.training.bases_lr = args.bases_lr
    
    window_size = cfg.data.trial_len

    # Get model paths
    if args.model_paths:
        # Parse custom model paths from command line
        model_paths = {}
        for model_spec in args.model_paths:
            if ':' not in model_spec:
                print(f"Error: Model path specification must be in format 'name:path', got: {model_spec}")
                sys.exit(1)
            name, path = model_spec.split(':', 1)
            model_paths[name] = Path(path).resolve()
            if not model_paths[name].exists():
                print(f"Error: Model path does not exist: {model_paths[name]}")
                sys.exit(1)
        print(f"Using custom model paths: {list(model_paths.keys())}")
    else:
        all_model_paths = get_model_paths()
        if args.models:
            model_paths = {k: v for k, v in all_model_paths.items() if k in args.models}
            if not model_paths:
                print(f"Error: None of the specified models found: {args.models}")
                print(f"Available models: {list(all_model_paths.keys())}")
                sys.exit(1)
        else:
            model_paths = all_model_paths

    print(f"Evaluating models: {list(model_paths.keys())}")

    # Load session information
    held_in_sessions_map, held_out_sessions_map = load_held_in_sessions(model_paths)
    shared_held_out_sessions = find_shared_held_out_sessions(held_out_sessions_map)

    # Select adaptation sessions
    adapt_session_ids = select_adapt_sessions(
        shared_held_out_sessions,
        args.max_adapt_sessions,
        args.adapt_session
    )

    # Process sessions in groups of 5 to manage memory
    SESSION_GROUP_SIZE = 5 if max(args.support_sizes) >= 2500 else 15
    session_groups = [
        adapt_session_ids[i:i + SESSION_GROUP_SIZE]
        for i in range(0, len(adapt_session_ids), SESSION_GROUP_SIZE)
    ]

    print(f"\nProcessing {len(adapt_session_ids)} sessions in {len(session_groups)} group(s) of up to {SESSION_GROUP_SIZE}")

    # Run TTA for each group and aggregate results
    all_results = []
    for group_idx, session_group in enumerate(session_groups):
        print(f"\n{'='*80}")
        print(f"Processing group {group_idx + 1}/{len(session_groups)}: {len(session_group)} sessions")
        print(f"{'='*80}\n")

        # Prepare data for this group
        data_train, data_test, embeddings_rest = prepare_data(
            session_group,
            window_size,
            args.batch_size_per_session,
            device
        )

        # Run TTA sweep for this group
        group_results = run_tta_sweep(
            model_paths,
            held_in_sessions_map,
            args.support_sizes,
            session_group,
            cfg,
            data_train,
            data_test,
            args.tta_epochs,
            device,
            args.output_dir,
            include_vanilla_tbfm=args.include_vanilla_tbfm,
            vanilla_tbfm_epochs=args.vanilla_tbfm_epochs,
            include_fresh_tbfm=args.include_fresh_tbfm,
            fresh_tbfm_epochs=args.fresh_tbfm_epochs,
            use_multi_gpu=args.use_multi_gpu,
            gpu_ids=args.gpu_ids,
        )

        all_results.append(group_results)

        # Clean up GPU memory between groups
        del data_train, data_test, embeddings_rest
        torch.cuda.empty_cache()

    # Aggregate results from all groups
    print(f"\n{'='*80}")
    print(f"Aggregating results from {len(session_groups)} group(s)")
    print(f"{'='*80}\n")

    # Combine results (use first group's metadata and baselines, merge TTA runs)
    results = all_results[0]
    for group_results in all_results[1:]:
        results["runs"].extend(group_results["runs"])
        # Merge grid results (dict of model -> strategy -> list of (support_size, r2))
        for model_key in group_results["grid"]:
            for strategy_key in group_results["grid"][model_key]:
                results["grid"][model_key][strategy_key].extend(
                    group_results["grid"][model_key][strategy_key]
                )

    # Update metadata to reflect all sessions
    results["metadata"]["adapt_session_ids"] = adapt_session_ids

    # Save and plot results
    save_results(results, args.output_dir)
    plot_results(results, args.output_dir, show=not args.no_plot_display)

    print("\n" + "=" * 80)
    print("TTA Evaluation Complete!")
    print("=" * 80)
    
    # Send completion notification
    try:
        notifications.notify_training_complete(
            model_name="TTA Sweep",
            metrics={},
            output_dir=str(args.output_dir)
        )
    except Exception as e:
        print(f"Failed to send notification: {e}")


if __name__ == "__main__":
    main()
