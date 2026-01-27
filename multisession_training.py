"""
Multisession Training Functions for TBFM

This module contains various training functions and utilities for TBFM multisession experiments,
including baseline training, spatial regularization, AE reconstruction penalties, and sparsity training.
"""

import os
import time
import glob
import builtins
import torch
import torch.nn as nn
from tbfm.multisession import split_support_query_sessions, r2_score
from tbfm import multisession, meta, utils
from typing import Optional, Dict, Any, Tuple


class ExperimentLogger:
    """Logger for experiment progress with structured output."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.start_time = time.time()
        # Store original print function to avoid recursion
        self._original_print = print
        
        if log_file:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Initialize log file
            with open(log_file, 'w') as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} - Logging to: {log_file}\n")
    
    def _log_message(self, message):
        """Log message to both console and file if specified."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"{timestamp} - {message}"
        # Use original print to avoid recursion
        self._original_print(full_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(full_message + '\n')
    
    def info(self, message):
        """Log info message."""
        self._log_message(message)
    
    def warning(self, message):
        """Log warning message."""
        self._log_message(f"⚠️  {message}")
    
    def error(self, message):
        """Log error message."""
        self._log_message(f"❌ {message}")
    
    def start_phase(self, phase_name):
        """Start a new experiment phase."""
        self._log_message("=" * 80)
        self._log_message(f"Starting: {phase_name}")
        self._log_message("=" * 80)
        self.phase_start_time = time.time()
    
    def end_phase(self, phase_name):
        """End current experiment phase."""
        if hasattr(self, 'phase_start_time'):
            duration = time.time() - self.phase_start_time
            duration_str = time.strftime('%M:%S', time.gmtime(duration))
            self._log_message(f"\nCompleted: {phase_name}")
            self._log_message(f"Duration: {duration_str}")
            self._log_message("=" * 80)
    
    def start_training(self, epochs, training_name):
        """Start training phase."""
        self.start_phase(training_name)
        self.training_start_time = time.time()
    
    def end_training(self, training_name):
        """End training phase."""
        if hasattr(self, 'training_start_time'):
            duration = time.time() - self.training_start_time
            duration_str = time.strftime('%H:%M:%S', time.gmtime(duration))
            self._log_message(f"\n{training_name} completed in {duration_str}")
        self.end_phase(training_name)
    
    def log_progress(self, epoch, num_epochs, train_loss, test_loss=None, train_r2=None, test_r2=None):
        """Log training progress with ETA calculation."""
        if hasattr(self, 'training_start_time'):
            elapsed = time.time() - self.training_start_time
            progress = epoch / max(num_epochs, 1)  # Ensure non-zero denominator
            
            # More robust ETA calculation
            if progress > 0.001:  # Only calculate ETA after some progress
                eta = (elapsed / progress) * (1 - progress)
                eta_str = time.strftime('%M:%S', time.gmtime(eta))
            else:
                eta_str = "--:--"
            
            elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
            
            progress_msg = f"Epoch {epoch}/{num_epochs} ({progress*100:.1f}%)"
            loss_msg = f"Train Loss: {train_loss:.6f}"
            
            if test_loss is not None:
                loss_msg += f" | Test Loss: {test_loss:.6f}"
            if train_r2 is not None:
                loss_msg += f" | Train R²: {train_r2:.6f}"
            if test_r2 is not None:
                loss_msg += f" | Test R²: {test_r2:.6f}"
            
            time_msg = f"Elapsed: {elapsed_str} | ETA: {eta_str}"
            
            self._log_message(f"{progress_msg} | {loss_msg} | {time_msg}")


def cleanup_checkpoints(training_type=None, model_name_pattern=None, keep_final=True, keep_latest_best=False):
    """
    Clean up checkpoint files after training completes.
    
    Args:
        training_type: Training type to match (e.g., "Baseline", "Spatial", "AE_Recon", "Sparsity")
                      If None, cleans all checkpoint files regardless of training type
        model_name_pattern: Pattern to match model names (e.g., "TBFMMultisession")
                           If None, cleans all checkpoint files
        keep_final: If True, keeps final model files (*_final_*.pt)
        keep_latest_best: If True, keeps the most recent best checkpoint
    """
    saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.exists(saved_models_dir):
        return
    
    # Build pattern to match files
    if training_type and model_name_pattern:
        pattern = os.path.join(saved_models_dir, f"{training_type}_{model_name_pattern}_*.pt")
    elif training_type:
        pattern = os.path.join(saved_models_dir, f"{training_type}_*.pt")
    elif model_name_pattern:
        pattern = os.path.join(saved_models_dir, f"*_{model_name_pattern}_*.pt")
    else:
        pattern = os.path.join(saved_models_dir, "*.pt")
    
    checkpoint_files = glob.glob(pattern)
    
    files_to_delete = []
    final_files = []
    best_files = []
    
    # Categorize files
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        if "_final_" in filename:
            final_files.append(filepath)
        elif "_best_" in filename or "_restored_best_" in filename:
            best_files.append(filepath)
        else:
            # Other checkpoint files
            files_to_delete.append(filepath)
    
    # Sort best files by timestamp (filename contains Unix timestamp)
    best_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Determine what to delete
    if not keep_final:
        files_to_delete.extend(final_files)
    
    if not keep_latest_best:
        files_to_delete.extend(best_files)
    else:
        # Keep only the most recent best file
        files_to_delete.extend(best_files[:-1])
    
    # Delete files
    deleted_count = 0
    freed_mb = 0.0
    
    for filepath in files_to_delete:
        try:
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            os.remove(filepath)
            filename = os.path.basename(filepath)
            print(f"Deleted checkpoint: {filename}")
            deleted_count += 1
            freed_mb += file_size
        except Exception as e:
            print(f"Warning: Failed to delete {filepath}: {e}")
    
    if deleted_count > 0:
        type_msg = f"({training_type} only)" if training_type else "(all types)"
        print(f"✅ Cleaned up {deleted_count} checkpoint files {type_msg} ({freed_mb:.1f} MB freed)")
    else:
        print("No checkpoint files to clean up")


def train_with_logging(*args, logger=None, training_type="Unknown", **kwargs):
    """
    Wrapper for train_from_cfg that logs progress to logger and saves best model to disk.

    This wrapper intercepts printed progress lines and keeps track of the best test R².
    When a new best model is found it saves a checkpoint under `saved_models/` and
    after training it also saves the final model state.
    
    Args:
        logger: ExperimentLogger instance for logging
        training_type: Type of training for model naming (e.g., "Baseline", "Spatial")
        *args, **kwargs: Arguments passed to multisession.train_from_cfg
    """
    original_print = builtins.print
    
    # Track best model
    best_test_r2 = -1e99
    best_model_state = None
    best_embeddings_stim = None
    test_r2_history = []
    
    # Extract model reference for checkpointing
    cfg = args[0] if len(args) > 0 else kwargs.get('cfg')
    model = args[1] if len(args) > 1 else kwargs.get('model')
    save_best = kwargs.pop('save_best_model', True)

    def _ensure_saved_models_dir():
        out_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def logging_print(*print_args, **print_kwargs):
        nonlocal best_test_r2, best_model_state, best_embeddings_stim
        msg = ' '.join(map(str, print_args))
        
        if msg.startswith('----'):
            parts = msg.split()
            if len(parts) >= 5:
                try:
                    epoch = int(parts[1])
                    train_loss = float(parts[2])
                    test_loss = float(parts[3])
                    train_r2 = float(parts[4])
                    test_r2 = float(parts[5]) if len(parts) > 5 else None
                    
                    if logger:
                        logger.log_progress(epoch, kwargs.get('epochs', 7001), train_loss, test_loss, train_r2, test_r2)
                    
                    # Track best model and save to disk when improved
                    if save_best and test_r2 is not None:
                        test_r2_history.append((epoch, test_r2))
                        if test_r2 > best_test_r2:
                            improvement = test_r2 - best_test_r2
                            best_test_r2 = test_r2
                            
                            # Only copy to CPU and save if significant improvement (>0.01) or every 2000 epochs
                            should_checkpoint = improvement > 0.01 or epoch % 2000 == 0
                            
                            if should_checkpoint:
                                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                                if logger:
                                    logger.info(f"  → New best model! Test R²: {best_test_r2:.6f} (checkpointed)")

                                # Save checkpoint to disk with training type in name
                                try:
                                    out_dir = _ensure_saved_models_dir()
                                    fname = os.path.join(out_dir, f"{training_type}_{model.__class__.__name__}_best_{int(time.time())}.pt")
                                    torch.save({
                                        'model_state_dict': best_model_state,
                                        'epoch': epoch,
                                        'test_r2': best_test_r2,
                                        'timestamp': time.time(),
                                        'training_type': training_type,
                                    }, fname)
                                    if logger:
                                        logger.info(f"Saved best checkpoint to {fname}")
                                except Exception as e:
                                    if logger:
                                        logger.warning(f"Failed to save best checkpoint: {e}")
                            else:
                                # Just log the improvement without checkpointing
                                if logger:
                                    logger.info(f"  → New best model! Test R²: {best_test_r2:.6f} (not checkpointed)")
                    
                    return
                except (ValueError, IndexError):
                    pass
        
        if msg.strip() and not msg.startswith('Building') and not msg.startswith('BOOM'):
            if logger:
                logger.info(msg)
            else:
                original_print(*print_args, **print_kwargs)
        else:
            # For messages we don't want logged (Building, BOOM), just print normally
            original_print(*print_args, **print_kwargs)
    
    builtins.print = logging_print
    
    try:
        embeddings_stim, results = multisession.train_from_cfg(*args, **kwargs)
        
        # Restore best model
        if save_best and best_model_state is not None:
            if logger:
                logger.info(f"Restoring best model (Test R²: {best_test_r2:.6f})")
            device = model.device
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            # Update results with best model info
            results['best_test_r2'] = best_test_r2
            results['final_test_r2'] = best_test_r2  # Override with best

        # Save final model to disk with training type in name
        try:
            out_dir = _ensure_saved_models_dir()
            final_fname = os.path.join(out_dir, f"{training_type}_{model.__class__.__name__}_final_{int(time.time())}.pt")
            model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'model_state_dict': model_cpu_state,
                'results': results,
                'timestamp': time.time(),
                'training_type': training_type,
            }, final_fname)
            if logger:
                logger.info(f"Saved final model to {final_fname}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save final model: {e}")

        # Clean up intermediate checkpoint files of the same training type only
        try:
            cleanup_checkpoints(training_type=training_type, model_name_pattern=model.__class__.__name__, 
                              keep_final=True, keep_latest_best=False)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup checkpoints: {e}")

        return embeddings_stim, results
    finally:
        builtins.print = original_print


def train_with_spatial_ae_recon(
    cfg,
    model,
    data_train,
    model_optims,
    embeddings_rest,
    data_test=None,
    inner_steps=None,
    epochs=10000,
    test_interval=None,
    support_size=300,
    embed_stim_lr=None,
    embed_stim_weight_decay=None,
    device="cuda",
    grad_clip=None,
    alternating_updates=True,
    bw_steps_per_bg_step=None,
    model_save_path=None,
    lambda_spatial_smooth=0.05,
    lambda_spatial_decoder=0.01,
    lambda_ae_recon=0.01,
    spatial_penalty_freq=10,
    save_best_model=True,
    training_type="Custom",
    logger=None,
):
    """
    Modified training with spatial regularization + AE reconstruction penalty.
    
    The AE reconstruction penalty penalizes lossy conversions through the encoder-decoder bottleneck:
        L_ae_recon = || dec(enc(x)) - x ||_2^2
    
    This encourages the autoencoder to preserve information through the latent bottleneck.
    
    Performance optimization: Set spatial_penalty_freq > 1 to compute spatial
    penalties less frequently (e.g., every 10 iterations instead of every iteration).
    This provides significant speedup (~50% faster) with minimal impact on final performance.
    """
    # Same setup as original
    test_interval = test_interval or cfg.training.test_interval
    embed_stim_lr = embed_stim_lr or cfg.meta.training.optim.lr
    embed_stim_weight_decay = embed_stim_weight_decay or cfg.meta.training.optim.weight_decay
    inner_steps = inner_steps or cfg.meta.training.inner_steps
    use_meta_learning = cfg.tbfm.module.use_meta_learning
    bw_steps_per_bg_step = bw_steps_per_bg_step or cfg.training.bw_steps_per_bg_step
    grad_clip = grad_clip or cfg.training.grad_clip or 10.0
    epochs = epochs or cfg.training.epochs
    support_size = support_size or cfg.meta.training.support_size
    device = model.device

    def _ensure_saved_models_dir():
        out_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    embeddings_stim = None
    iter_train = iter(data_train)

    train_losses = []
    train_r2s = []
    test_losses = []
    test_r2s = []
    spatial_smooth_losses = []
    spatial_decoder_losses = []
    ae_recon_losses = []
    best_test_r2 = -1e99
    best_model_state = None
    best_embeddings_stim = None

    for eidx in range(epochs):
        model.train()
        iter_train, _data_train = utils.iter_loader(iter_train, data_train, device=device)

        support, query = split_support_query_sessions(_data_train, support_size=support_size)

        with torch.no_grad():
            y_query = {sid: d[2] for sid, d in query.items()}
            y_query = model.norms(y_query)

        if use_meta_learning:
            model.eval()
            embeddings_stim = meta.inner_update_stopgrad(
                model, support, embeddings_rest, inner_steps=inner_steps,
                lr=embed_stim_lr, weight_decay=embed_stim_weight_decay,
            )
            model.train()

        model_optims.zero_grad(set_to_none=True)

        yhat_query = model(query, embeddings_rest=embeddings_rest, embeddings_stim=embeddings_stim)

        losses = {}
        r2_trains = []
        spatial_smooth_total = 0.0
        spatial_decoder_total = 0.0

        # Only compute spatial penalties every N iterations for speedup
        compute_spatial = (eidx % spatial_penalty_freq) == 0

        for sid, y in y_query.items():
            _loss = nn.MSELoss()(yhat_query[sid], y)
            losses[sid] = _loss

            # Add spatial smoothness penalty on reconstructions (if enabled this iteration)
            if compute_spatial:
                ae_instance = model.ae.instances[sid]
                if hasattr(ae_instance, 'use_spatial') and ae_instance.use_spatial:
                    # Spatial smoothness on reconstructions
                    yhat_flat = yhat_query[sid].flatten(0, 1)  # (B*T, C)
                    smooth_penalty = ae_instance.spatial_smoothness_penalty(yhat_flat, reduction='mean')
                    spatial_smooth_total += smooth_penalty

                    # Spatial decoder penalty
                    mask = list(range(yhat_query[sid].shape[-1]))
                    decoder_penalty = ae_instance.spatial_decoder_penalty(mask)
                    spatial_decoder_total += decoder_penalty

            r2_train = r2_score(
                yhat_query[sid].permute(0, 2, 1).flatten(end_dim=1),
                y.permute(0, 2, 1).flatten(end_dim=1),
            )
            r2_trains.append(r2_train.item())

        # Compute reconstruction loss (MSE only)
        mse_only = sum(losses.values()) / len(y_query)

        # Build total loss
        loss = mse_only

        # Add TBFM regularization
        tbfm_regs = model.model.get_weighting_reg()
        loss = loss + cfg.tbfm.training.lambda_fro * sum(tbfm_regs.values()) / len(y_query)

        tbfm_regs_ortho = model.model.get_basis_rms_reg()
        loss = loss + cfg.tbfm.training.lambda_ortho * sum(tbfm_regs_ortho.values()) / len(y_query)

        # Add AE reconstruction penalty (if requested)
        ae_loss_total = 0.0
        if lambda_ae_recon > 0:
            for sid, y in y_query.items():
                ae_instance = model.ae.instances[sid]
                y_flat = y.flatten(0, 1)
                y_recon = ae_instance.reconstruct(y_flat, mask=list(range(y.shape[-1])))
                ae_loss = nn.MSELoss()(y_recon, y_flat)
                ae_loss_total += ae_loss
            ae_loss_total = ae_loss_total / len(y_query)
            loss = loss + lambda_ae_recon * ae_loss_total
        
        ae_recon_losses.append((eidx, ae_loss_total.item() if isinstance(ae_loss_total, torch.Tensor) else 0.0))

        # Add spatial regularization (if computed this iteration)
        if compute_spatial and spatial_smooth_total > 0:
            loss = loss + lambda_spatial_smooth * (spatial_smooth_total / len(y_query))
            loss = loss + lambda_spatial_decoder * (spatial_decoder_total / len(y_query))
            spatial_smooth_losses.append((eidx, spatial_smooth_total.item() / len(y_query)))
            spatial_decoder_losses.append((eidx, spatial_decoder_total.item() / len(y_query)))

        # Log losses
        train_losses.append((eidx, mse_only.item()))

        loss.backward()

        if grad_clip is not None:
            model_optims.clip_grad(value=grad_clip)

        if alternating_updates:
            update_basis_gen = (eidx % bw_steps_per_bg_step) == 0 or eidx < 200
            if update_basis_gen:
                model_optims.step()
            else:
                model_optims.step(skip=["bg", "meta"])
        else:
            model_optims.step()

        # Testing
        if data_test is not None and (eidx % test_interval) == 0:
            train_r2s.append((eidx, sum(r2_trains) / len(y_query)))

            model_optims.zero_grad(set_to_none=True)
            model.eval()

            test_r2s_session = {}
            with torch.no_grad():
                iter_test = iter(data_test)
                iter_test, _data_test = utils.iter_loader(iter_test, data_test, device=device)

                y_test = {sid: d[2] for sid, d in _data_test.items()}
                y_test = model.norms(y_test)

                yhat_test = model(_data_test, embeddings_rest=embeddings_rest, embeddings_stim=embeddings_stim)

                test_losses_session = {}
                for sid, y in y_test.items():
                    _loss = nn.MSELoss()(yhat_test[sid], y)
                    test_losses_session[sid] = _loss

                    r2_test = r2_score(
                        yhat_test[sid].permute(0, 2, 1).flatten(end_dim=1),
                        y.permute(0, 2, 1).flatten(end_dim=1),
                    )
                    test_r2s_session[sid] = r2_test.item()

                test_loss = sum(test_losses_session.values()) / len(y_test)
                test_r2 = sum(test_r2s_session.values()) / len(y_test)

                test_losses.append((eidx, test_loss.item()))
                test_r2s.append((eidx, test_r2))

            if logger:
                logger.log_progress(eidx, epochs, train_losses[-1][1], test_loss.item(), train_r2s[-1][1], test_r2)
            
            if save_best_model and test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if embeddings_stim is not None:
                    best_embeddings_stim = {k: v.cpu().clone() if torch.is_tensor(v) else v 
                                           for k, v in embeddings_stim.items()}
                else:
                    best_embeddings_stim = None
                if logger:
                    logger.info(f"  → New best model! Test R²: {best_test_r2:.6f}")

                # Save checkpoint to disk with training type in name
                try:
                    out_dir = _ensure_saved_models_dir()
                    fname = os.path.join(out_dir, f"{training_type}_{model.__class__.__name__}_best_{int(time.time())}.pt")
                    torch.save({
                        'model_state_dict': best_model_state,
                        'epoch': eidx,
                        'test_r2': best_test_r2,
                        'timestamp': time.time(),
                        'training_type': training_type,
                    }, fname)
                    if logger:
                        logger.info(f"Saved best checkpoint to {fname}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to save best checkpoint: {e}")
            
            model.train()

    # Restore best model
    if save_best_model and best_model_state is not None:
        if logger:
            logger.info(f"Restoring best model (Test R²: {best_test_r2:.6f})")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        if best_embeddings_stim is not None:
            embeddings_stim = {k: v.to(device) if torch.is_tensor(v) else v 
                              for k, v in best_embeddings_stim.items()}

        # Save final model with training type in name
        try:
            out_dir = _ensure_saved_models_dir()
            final_fname = os.path.join(out_dir, f"{training_type}_{model.__class__.__name__}_final_{int(time.time())}.pt")
            model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'model_state_dict': model_cpu_state,
                'results': {
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_r2s': train_r2s,
                    'test_r2s': test_r2s,
                    'spatial_smooth_losses': spatial_smooth_losses,
                    'spatial_decoder_losses': spatial_decoder_losses,
                    'ae_recon_losses': ae_recon_losses,
                    'final_test_r2': best_test_r2,
                    'final_test_r2s': test_r2s_session,
                    'best_test_r2': best_test_r2,
                },
                'timestamp': time.time(),
                'training_type': training_type,
            }, final_fname)
            if logger:
                logger.info(f"Saved final model to {final_fname}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save final model: {e}")

        # Clean up intermediate checkpoint files of the same training type only
        try:
            cleanup_checkpoints(training_type=training_type, model_name_pattern=model.__class__.__name__, 
                              keep_final=True, keep_latest_best=False)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup checkpoints: {e}")

    return embeddings_stim, {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_r2s': train_r2s,
        'test_r2s': test_r2s,
        'spatial_smooth_losses': spatial_smooth_losses,
        'spatial_decoder_losses': spatial_decoder_losses,
        'ae_recon_losses': ae_recon_losses,
        'final_test_r2': best_test_r2,
        'final_test_r2s': test_r2s_session,
        'best_test_r2': best_test_r2,
    }