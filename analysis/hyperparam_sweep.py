#!/usr/bin/env python3
"""
Hyperparameter Sweep Launcher
Uses Python multiprocessing for better control over parallel GPU jobs
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
import multiprocessing as mp
from queue import Empty

# Configuration
NUM_BASES = 100
NUM_SESSIONS = 25  # Full sweep when all data is uploaded (currently only 10 available)
TRAIN_SIZE = 5000
EPOCHS = 12001
COADAPT = True

# GPU allocation
GPUS = [0, 1, 2, 3]

# Hyperparameter grid
BATCH_SIZES_PER_SESSION = [250, 500, 750]
LATENT_DIMS = [60, 85, 110]
BASIS_RESIDUAL_RANKS = [8, 16, 32]
RESIDUAL_MLP_HIDDENS = [16, 32, 64]

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def log_message(color, message):
    """Print colored message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {message}{NC}", flush=True)


def run_experiment(gpu_id, config, sweep_dir, lock):
    """
    Run a single experiment on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        config: Tuple of (batch_size_per_session, latent_dim, residual_rank, mlp_hidden)
        sweep_dir: Output directory
        lock: Multiprocessing lock for synchronized logging
    """
    batch_size, latent_dim, residual_rank, mlp_hidden = config
    total_batch = batch_size * NUM_SESSIONS
    exp_name = f"bs{total_batch}_ld{latent_dim}_rr{residual_rank}_mlp{mlp_hidden}"
    out_dir = sweep_dir / exp_name
    
    with lock:
        log_message(CYAN, f"[GPU {gpu_id}] Starting: {exp_name}")
    
    # Build command
    cmd = [
        "python", "tma_standalone.py",
        str(NUM_BASES),
        str(NUM_SESSIONS),
        str(gpu_id),
        "true" if COADAPT else "false",
        str(residual_rank),
        str(TRAIN_SIZE),
        "false",
        "--latent-dim", str(latent_dim),
        "--batch-size-per-session", str(batch_size),
        "--residual-mlp-hidden", str(mlp_hidden),
        "--out-dir", str(out_dir),
    ]
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # TBFM_DATA_DIR should already be set in environment
    
    # Run experiment
    log_file = sweep_dir / f"{exp_name}.log"
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd(),
            )
        
        success = result.returncode == 0
        
        with lock:
            if success:
                log_message(GREEN, f"[GPU {gpu_id}] ✓ Completed: {exp_name}")
                status = "SUCCESS"
            else:
                log_message(RED, f"[GPU {gpu_id}] ✗ Failed: {exp_name}")
                status = "FAILED"
            
            # Append to results CSV
            with open(sweep_dir / "results.csv", "a") as rf:
                rf.write(f"{exp_name},{status},{datetime.now()}\n")
        
        return success
        
    except Exception as e:
        with lock:
            log_message(RED, f"[GPU {gpu_id}] ✗ Error in {exp_name}: {e}")
            with open(sweep_dir / "results.csv", "a") as rf:
                rf.write(f"{exp_name},ERROR,{datetime.now()}\n")
        return False


def gpu_worker(gpu_id, job_queue, result_queue, sweep_dir, lock):
    """
    Worker process for a single GPU.
    
    Args:
        gpu_id: GPU device ID
        job_queue: Queue of jobs (configs) to process
        result_queue: Queue to put results
        sweep_dir: Output directory
        lock: Multiprocessing lock
    """
    # Staggered delay to avoid simultaneous PyTorch imports
    # This prevents library loading race conditions
    time.sleep(gpu_id * 3.0)  # Increased to 3 seconds per GPU
    
    while True:
        try:
            config = job_queue.get(timeout=1)
            if config is None:  # Poison pill
                break
            
            success = run_experiment(gpu_id, config, sweep_dir, lock)
            result_queue.put((config, success))
            
        except Empty:
            break
        except Exception as e:
            with lock:
                log_message(RED, f"[GPU {gpu_id}] Worker error: {e}")
            break


def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(f"hyperparam_sweep_{timestamp}")
    sweep_dir.mkdir(exist_ok=True)
    
    log_message(BLUE, "=" * 80)
    log_message(BLUE, "Hyperparameter Sweep - Cloudbank (Python Launcher)")
    log_message(BLUE, "=" * 80)
    print()
    log_message(YELLOW, "Configuration:")
    log_message(NC, f"  Fixed Parameters:")
    log_message(NC, f"    - Num Bases: {NUM_BASES}")
    log_message(NC, f"    - Num Sessions: {NUM_SESSIONS}")
    log_message(NC, f"    - Train Size: {TRAIN_SIZE}")
    log_message(NC, f"    - Epochs: {EPOCHS}")
    log_message(NC, f"    - Co-adapt: {COADAPT}")
    print()
    log_message(NC, f"  Hyperparameter Grid:")
    log_message(NC, f"    - Batch Sizes (per session): {BATCH_SIZES_PER_SESSION}")
    log_message(NC, f"    - Latent Dims: {LATENT_DIMS}")
    log_message(NC, f"    - Basis Residual Ranks: {BASIS_RESIDUAL_RANKS}")
    log_message(NC, f"    - MLP Hidden Dims: {RESIDUAL_MLP_HIDDENS}")
    print()
    
    # Generate all configurations
    configs = list(product(
        BATCH_SIZES_PER_SESSION,
        LATENT_DIMS,
        BASIS_RESIDUAL_RANKS,
        RESIDUAL_MLP_HIDDENS
    ))
    
    total_experiments = len(configs)
    log_message(YELLOW, f"Total experiments: {total_experiments}")
    log_message(YELLOW, f"Using {len(GPUS)} GPUs: {GPUS}")
    log_message(YELLOW, f"Output directory: {sweep_dir}")
    print()
    
    # Create results CSV
    with open(sweep_dir / "results.csv", "w") as f:
        f.write("experiment,status,timestamp\n")
    
    # Create job queue and result queue
    manager = mp.Manager()
    job_queue = manager.Queue()
    result_queue = manager.Queue()
    lock = manager.Lock()
    
    # Populate job queue
    for config in configs:
        job_queue.put(config)
    
    # Add poison pills
    for _ in GPUS:
        job_queue.put(None)
    
    # Start worker processes
    processes = []
    for gpu_id in GPUS:
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, job_queue, result_queue, sweep_dir, lock)
        )
        p.start()
        processes.append(p)
    
    # Monitor progress
    completed = 0
    successes = 0
    failures = 0
    
    start_time = time.time()
    
    while completed < total_experiments:
        try:
            config, success = result_queue.get(timeout=1)
            completed += 1
            if success:
                successes += 1
            else:
                failures += 1
            
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_experiments - completed) / rate if rate > 0 else 0
            
            log_message(
                YELLOW,
                f"Progress: {completed}/{total_experiments} "
                f"({successes} ✓, {failures} ✗) "
                f"[{rate:.2f} exp/min, ETA: {eta/60:.1f} min]"
            )
            
        except Empty:
            # Check if all processes are done
            if all(not p.is_alive() for p in processes):
                break
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    elapsed_total = time.time() - start_time
    
    log_message(GREEN, "=" * 80)
    log_message(GREEN, "Sweep Complete!")
    log_message(GREEN, "=" * 80)
    log_message(NC, f"Results saved to: {sweep_dir}")
    log_message(NC, f"Total time: {elapsed_total/3600:.2f} hours")
    print()
    log_message(YELLOW, "Summary:")
    log_message(GREEN, f"  Successful: {successes}")
    log_message(RED, f"  Failed: {failures}")
    print()
    log_message(NC, "To analyze results, run:")
    log_message(NC, f"  python analyze_hyperparam_sweep.py {sweep_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
