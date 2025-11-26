import math
import random
import os
import sys

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torcheval.metrics.functional import r2_score
from typing import Dict, Tuple, List

from . import ae
from . import dataset
from . import meta
from . import flow_ae  # Normalizing flow autoencoder support
from . import normalizers
from . import tbfm
from . import utils

from ._multisession_module import TBFMMultisession


def build_from_cfg(
    cfg,
    session_data,
    shared_ae=False,
    latent_dim=None,
    base_model_path=None,
    quiet=False,
    device=None,
):
    """
    Build complete multisession TBFM model from config.

    Autoencoder type is determined by cfg.ae.module._target_:
    - tbfm.ae.LinearChannelAE: Linear tied-weight autoencoder (default)
    - tbfm.flow_ae.FlowChannelAE: Normalizing flow autoencoder (invertible)

    To use flow autoencoder, set: ae=flow in config or overrides.
    """
    latent_dim = cfg.latent_dim

    with torch.no_grad():
        # Normalizers ------
        if not quiet:
            print("Building and fitting normalizers...")
        norms = normalizers.from_cfg(cfg, session_data, device=device)

        # Autoencoders ------
        # Note: ae.from_cfg_and_data uses Hydra instantiate, which respects
        # cfg.ae.module._target_ to determine which autoencoder class to create.
        # This allows seamless switching between LinearChannelAE and FlowChannelAE.
        if not quiet:
            print("Building and warm starting AEs...")
        aes = ae.from_cfg_and_data(
            cfg,
            session_data,
            use_lora=shared_ae,
            latent_dim=latent_dim,
            shared=shared_ae,
            device=device,
        )

        # TBFM ------
        if not base_model_path:
            if not quiet:
                print("Building TBFM...")
            _tbfm = tbfm.from_cfg(
                cfg,
                tuple(session_data.keys()),
                shared=cfg.tbfm.sharing.shared,
                device=device,
            )
        else:
            if not quiet:
                print("Loading base TBFM from file...")
            _tbfm = tbfm.shared_from_cfg_and_base(
                cfg,
                tuple(session_data.keys()),
                base_model_path,
                device=device,
            )

        # Cleared for takeoff ------
        if not quiet:
            print("BOOM! Dino DNA!")
    return TBFMMultisession(norms, aes, _tbfm, device=device)


def save_model(model, path, tbfm_only=True):
    # We save only the TBFM for now, since that is how we are doing TTA
    if not tbfm_only:
        raise NotImplementedError()
    else:
        instances = set(model.model.instances.values())
        if len(instances) != 1:
            raise NotImplementedError()

        model_tbfm = next(iter(instances))
        torch.save(model_tbfm.state_dict(), path)


def get_optims(cfg, model_ms: TBFMMultisession, embeddings_stim=None):
    optims_norms = tuple()
    if cfg.normalizers.training.coadapt:
        raise NotImplementedError("No normalizer adaptation yet")

    if cfg.ae.training.coadapt:
        use_two_stage = cfg.ae.use_two_stage

        if use_two_stage:
            # Two-stage AE returns a single optimizer with param groups
            optim_ae = model_ms.ae.get_optim(
                adapter_lr=cfg.ae.two_stage.adapter_lr,
                encoder_lr=cfg.ae.two_stage.encoder_lr,
                eps=cfg.ae.training.optim.eps,
                weight_decay=cfg.ae.training.optim.weight_decay,
                amsgrad=cfg.ae.training.optim.amsgrad,
            )
            optims_aes = [optim_ae]
        else:
            # Single-stage AE dispatcher returns dict of optimizers
            optims_aes = list(model_ms.ae.get_optim(**cfg.ae.training.optim).values())
    else:
        optims_aes = []

    # Will have only 1 elem if tbfm is shared.
    optims_model = model_ms.model.get_optim_custom_base(**cfg.tbfm.training.optim)

    def warmup_cos(warmup, cos=True):
        def _inner(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            if cos:
                t = (step - warmup) / max(1, cfg.training.epochs - warmup)
                return 0.5 * (1 + math.cos(math.pi * t))  # → goes to ~0
            return 1.0

        return _inner

    # Collect optimizers and schedulers by group
    bw_optims = []
    bw_schedulers = []
    bg_optims = []
    bg_schedulers = []
    meta_optims = []

    for sid, optims in optims_model.items():
        optim_bw, optim_bg, optim_meta = optims

        bw_optims.append(optim_bw)
        bw_schedulers.append(LambdaLR(optim_bw, lr_lambda=warmup_cos(1500, cos=False)))

        bg_optims.append(optim_bg)
        bg_schedulers.append(LambdaLR(optim_bg, lr_lambda=warmup_cos(5000)))

        if optim_meta is not None:
            meta_optims.append(optim_meta)

    # Add embeddings optimizer if co-adapting
    embed_optims = []
    if embeddings_stim is not None:
        embed_lr = cfg.meta.training.optim.lr
        embed_wd = cfg.meta.training.optim.weight_decay
        embed_optim = torch.optim.AdamW(
            list(embeddings_stim.values()), 
            lr=embed_lr, 
            weight_decay=embed_wd
        )
        embed_optims.append(embed_optim)

    # Create named optimizer groups
    groups = {
        "norm": {"optimizers": optims_norms, "schedulers": []},
        "ae": {"optimizers": optims_aes, "schedulers": []},
        "bw": {"optimizers": bw_optims, "schedulers": bw_schedulers},
        "bg": {"optimizers": bg_optims, "schedulers": bg_schedulers},
        "meta": {"optimizers": meta_optims, "schedulers": []},
        "embed": {"optimizers": embed_optims, "schedulers": []},
    }

    return utils.OptimCollection(groups)


# Training ------------------------------------------------------


def split_support_query_sessions(
    data_train,
    support_size: int,
):
    support = {}
    query = {}
    for session_id, d in data_train.items():
        support[session_id] = tuple(dd[:support_size] for dd in d)
        query[session_id] = tuple(dd[support_size:] for dd in d)

    return support, query


def train_from_cfg(
    cfg,
    model: nn.Module,
    data_train,  # yields per-session batches (stiminds, runways, y)
    model_optims,
    embeddings_rest,
    embeddings_stim=None,  # Pre-initialized embeddings for co-adaptation
    data_test=None,
    epochs: int = 10000,
    test_interval: int | None = None,
    support_size: int = 300,
    device: str = "cuda",
    grad_clip: float | None = None,
    alternating_updates: bool = True,  # Use alternating basis weight / basis gen updates
    bw_steps_per_bg_step: (
        int | None
    ) = None,  # How many basis weight updates per basis update
    embed_steps_per_other_step: int = 5,  # How many other updates per embedding update
    model_save_path: str | None = None,
):
    """
    One epoch over sessions with:
      - inner loop on support to adapt stim_embed (per-episode latent) OR co-adaptation
      - outer update on query for shared params
      - slow AE updates every step (small lr)
      - optional alternating updates: update basis weights more frequently than basis generator
      - optional co-adaptation of embeddings instead of inner loop

    Alternating updates rationale:
      - Basis weights have large gradients and adapt quickly
      - Basis generator (Bases) changes more slowly
      - By updating basis weights N times per basis generator update, we let the
        weighting layer stabilize to the current bases before changing the bases
      - When co-adapting embeddings, alternate between embeddings and other parameters

    Notes:
      • To add EMA for AE: keep a shadow copy of AE params and update with EMA after each optimizer.step().
    """
    # cfg overrides
    test_interval = test_interval or cfg.training.test_interval
    use_meta = cfg.tbfm.module.use_meta_learning
    coadapt_embeddings = cfg.meta.training.coadapt if use_meta else False
    embed_steps_per_other_step = embed_steps_per_other_step or cfg.meta.training.get('embed_steps_per_other_step', 5)
    bw_steps_per_bg_step = bw_steps_per_bg_step or cfg.training.bw_steps_per_bg_step
    grad_clip = grad_clip or cfg.training.grad_clip or 10.0
    epochs = epochs or cfg.training.epochs
    support_size = support_size or cfg.meta.training.support_size
    ae_freeze_epoch = cfg.ae.training.ae_freeze_epoch
    lambda_ae_recon = cfg.ae.training.lambda_ae_recon
    lambda_mu = cfg.ae.two_stage.lambda_mu
    lambda_cov = cfg.ae.two_stage.lambda_cov
    device = model.device

    # Initialize embeddings_stim as trainable parameters if co-adapting
    if embeddings_stim is None and use_meta and coadapt_embeddings:
        embed_dim_stim = model.model.bases.embed_dim_stim
        embeddings_stim = {}
        for session_id in embeddings_rest.keys():
            emb = torch.randn(embed_dim_stim, device=device) * 0.1
            emb.requires_grad = True
            embeddings_stim[session_id] = emb
    embeddings_stim = None  # default
    if use_meta:
        embed_dim_stim = model.model.bases.embed_dim_stim
        embeddings_stim = {}
        for session_id in embeddings_rest.keys():
            emb = torch.randn(embed_dim_stim, device=device) * 0.1
            emb.requires_grad = True
            embeddings_stim[session_id] = emb

    iter_train = iter(data_train)

    train_losses = []
    train_r2s = []
    test_losses = []
    test_r2s = []
    min_test_r2 = 1e99

    # Track outlier statistics
    outlier_stats = {
        "train_filtered_per_epoch": [],
        "test_filtered_per_epoch": [],
    }

    for eidx in range(epochs):
        model.train()
        iter_train, _data_train = utils.iter_loader(
            iter_train, data_train, device=device
        )

        # Apply outlier filtering to training data if enabled
        _data_train, filter_stats = utils.filter_batch_outliers(
            _data_train, model.norms, cfg
        )

        # Log filtering statistics occasionally
        if cfg.training.use_outlier_filtering and eidx % 100 == 0:
            outlier_stats["train_filtered_per_epoch"].append(
                (eidx, filter_stats["total_kept"], filter_stats["total_samples"])
            )

        # split into support/query
        support, query = split_support_query_sessions(
            _data_train,
            support_size=support_size,
        )

        with torch.no_grad():
            y_query = {sid: d[2] for sid, d in query.items()}
            y_query = model.norms(y_query)

        # ----- inner adaptation on support -----
        use_hypernetwork = cfg.meta.use_hypernetwork if hasattr(cfg.meta, 'use_hypernetwork') else False
        support_contexts = None

        if use_meta and not coadapt_embeddings:
            model.eval()
            if use_hypernetwork:
                # Hypernetwork mode: encode support set to contexts (single forward pass)
                support_contexts = meta.hypernetwork_adapt(
                    model,
                    support,
                    embeddings_rest,
                    cfg,
                )
                embeddings_stim = None  # Not used in hypernetwork mode
            else:
                # MAML mode: optimize embeddings via gradient descent
                embeddings_stim = meta.inner_update_stopgrad(
                    model,
                    support,
                    embeddings_rest,
                    cfg,
                )
            model.train()

        model_optims.zero_grad(set_to_none=True)

        yhat_query = model(
            query,
            embeddings_rest=embeddings_rest,
            embeddings_stim=embeddings_stim,
            support_contexts=support_contexts,
        )

        losses = {}
        r2_trains = []
        for sid, y in y_query.items():
            _loss = nn.MSELoss()(yhat_query[sid], y)
            losses[sid] = _loss

            yhat_flat = yhat_query[sid].permute(0, 2, 1).flatten(end_dim=1)
            y_flat = y.permute(0, 2, 1).flatten(end_dim=1)
            if yhat_flat.shape[0] >= 2:
                r2_train = r2_score(yhat_flat, y_flat)
                r2_trains.append(r2_train.item())
            else:
                r2_trains.append(0.0)

        cur_loss = sum(losses.values()) / len(y_query)
        train_losses.append((eidx, cur_loss.item()))

        # Add AE reconstruction loss (optional)
        use_two_stage = cfg.ae.use_two_stage

        if lambda_ae_recon > 0:
            runways_normalized, runways_recon = model.forward_reconstruct(query)
            ae_recon_loss = 0.0
            for sid, rn in runways_normalized.items():
                rr = runways_recon[sid]
                ae_recon_loss += nn.functional.mse_loss(rr, rn)
            ae_recon_loss /= len(query)

            cur_loss += lambda_ae_recon * ae_recon_loss
        else:
            runways_normalized = None

        # Add moment matching losses for two-stage AE (optional)
        if use_two_stage:
            if lambda_mu > 0 or lambda_cov > 0:
                # Normalize runways if not already done
                if runways_normalized is None:
                    runways = {sid: d[0] for sid, d in query.items()}
                    runways_normalized = model.norms(runways)

                # Collect latents from all query sessions
                all_z = []
                for sid, runway_norm in runways_normalized.items():
                    z = model.ae.encode(runway_norm, session_id=sid)
                    all_z.append(z.flatten(end_dim=-2))  # Flatten batch/time

                z_batch = torch.cat(all_z, dim=0)  # [total_samples, latent_dim]
                L_mu, L_cov = model.ae.moment_matching_loss(z_batch)

                if lambda_mu > 0:
                    cur_loss += lambda_mu * L_mu
                if lambda_cov > 0:
                    cur_loss += lambda_cov * L_cov

        tbfm_regs = model.model.get_weighting_reg()
        cur_loss += (
            cfg.tbfm.training.lambda_fro * sum(tbfm_regs.values()) / len(y_query)
        )

        tbfm_regs_ortho = model.model.get_basis_rms_reg()
        cur_loss += (
            cfg.tbfm.training.lambda_ortho
            * sum(tbfm_regs_ortho.values())
            / len(y_query)
        )

        loss = cur_loss

        loss.backward()

        if grad_clip is not None:
            model_optims.clip_grad(value=grad_clip)

        # Alternating updates: update basis weights more frequently than basis generator
        # Also: freeze AE after specified epoch to prevent late-stage overfitting
        skip_groups = []

        use_two_stage = cfg.ae.use_two_stage

        if ae_freeze_epoch is not None and eidx >= ae_freeze_epoch:
            if use_two_stage:
                # For two-stage: optionally freeze only shared encoder/decoder, keep adapters trainable
                freeze_only_shared = cfg.ae.two_stage.freeze_only_shared

                if freeze_only_shared:
                    # Freeze shared encoder/decoder parameters
                    for param in model.ae.encoder.parameters():
                        param.requires_grad = False
                    # Adapters remain trainable - don't skip "ae" group
                else:
                    # Freeze entire AE (adapters + encoder/decoder)
                    skip_groups.append("ae")
            else:
                # Single-stage: freeze all AE instances
                skip_groups.append("ae")

        if alternating_updates:
            # Every basis_weight_steps_per_basis iterations, update both
            # Otherwise, only update basis weights
            update_basis_gen = (eidx % bw_steps_per_bg_step) == 0 or eidx < 200

            if coadapt_embeddings:
                # When co-adapting embeddings, alternate between embeddings and other params
                update_embeddings = (eidx % embed_steps_per_other_step) == 0 or eidx < 200
                
                if update_embeddings and update_basis_gen:
                    # Update everything minus frozen groups
                    model_optims.step(skip=skip_groups)
                elif update_embeddings:
                    # Update embeddings and basis weights, skip basis gen and meta
                    model_optims.step(skip=["bg", "meta"] + skip_groups)
                elif update_basis_gen:
                    # Update basis gen and meta, skip embeddings
                    model_optims.step(skip=["embed"] + skip_groups)
                else:
                    # Update basis weights only, skip basis gen, meta, and embeddings
                    model_optims.step(skip=["bg", "meta", "embed"] + skip_groups)
            else:
                if update_basis_gen:
                    # Update everything (basis weights, basis gen, meta, ae, norm) minus frozen groups
                    model_optims.step(skip=skip_groups)
                else:
                    # Update only basis weights (skip basis gen and meta) minus frozen groups
                    model_optims.step(skip=["bg", "meta"] + skip_groups)
        else:
            model_optims.step(skip=skip_groups)

        if data_test is not None and (eidx % test_interval) == 0:
            train_r2s.append((eidx, sum(r2_trains) / len(y_query)))

            model_optims.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.eval()
                test_results = utils.evaluate_test_batches(
                    model,
                    data_test,
                    embeddings_rest,
                    embeddings_stim,
                    model.norms,
                    cfg,
                    device,
                    track_per_session_r2=False,
                    support_contexts=support_contexts,
                )

                loss = test_results["loss"]
                r2_test = test_results["r2"]
                test_losses.append((eidx, loss))
                test_r2s.append((eidx, r2_test))
                print(
                    "----", eidx, train_losses[-1][-1], loss, train_r2s[-1][-1], r2_test
                )

                if r2_test < min_test_r2:
                    min_test_r2 = r2_test
                    if model_save_path:
                        save_model(model, model_save_path, tbfm_only=True)

    # ----- (optional) EMA of AE params -----
    # for p, p_ema in zip(model.ae_parameters(), ae_ema_params):
    #     p_ema.data.mul_(1 - ema_alpha).add_(p.data, alpha=ema_alpha)

    use_meta = cfg.tbfm.module.use_meta_learning
    use_hypernetwork = cfg.meta.use_hypernetwork if hasattr(cfg.meta, 'use_hypernetwork') else False

    if use_meta:
        iter_train, _data_train = utils.iter_loader(
            iter_train, data_train, device=device
        )
        # split into support/query
        support, _ = split_support_query_sessions(
            _data_train,
            support_size=support_size,
        )

        model_optims.zero_grad(set_to_none=True)

        if use_hypernetwork:
            # Final hypernetwork encoding
            support_contexts = meta.hypernetwork_adapt(
                model,
                support,
                embeddings_rest,
                cfg,
            )
            embeddings_stim = None
        else:
            # Final MAML adaptation
            embeddings_stim = meta.inner_update_stopgrad(
                model,
                support,
                embeddings_rest,
                cfg,
                inner_steps=(3 * cfg.meta.training.inner_steps),
            )
            support_contexts = None

    # Final test evaluation
    model_optims.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.eval()
        final_test_results = utils.evaluate_test_batches(
            model,
            data_test,
            embeddings_rest,
            embeddings_stim,
            model.norms,
            cfg,
            device,
            track_per_session_r2=True,
            support_contexts=support_contexts,
        )

        loss = final_test_results["loss"]
        r2_test = final_test_results["r2"]
        final_test_r2s = final_test_results["per_session_r2"]
        y_hat_test = final_test_results["y_hat"]
        test_batch = final_test_results["y_test"]

        # Log outlier filtering stats for test set
        if final_test_results["outlier_stats"] is not None:
            outlier_stats["test_filtered_per_epoch"].append(
                (
                    epochs - 1,
                    final_test_results["outlier_stats"]["total_kept"],
                    final_test_results["outlier_stats"]["total_samples"],
                )
            )
            print(
                f"Test outlier filtering: kept {final_test_results['outlier_stats']['total_kept']}/"
                f"{final_test_results['outlier_stats']['total_samples']} "
                f"({final_test_results['outlier_stats']['pct_kept']:.1f}%)"
            )

        if r2_test < min_test_r2:
            min_test_r2 = r2_test
            if model_save_path:
                save_model(model, model_save_path, tbfm_only=True)

    print("Final:", loss, r2_test)

    results = {}
    results["train_losses"] = train_losses
    results["test_losses"] = test_losses
    results["train_r2s"] = train_r2s
    results["test_r2s"] = test_r2s
    results["y_hat"] = yhat_query
    results["outlier_stats"] = (
        outlier_stats if cfg.training.use_outlier_filtering else None
    )
    results["y"] = y_query
    results["y_hat_test"] = y_hat_test
    results["y_test"] = test_batch
    results["final_test_r2"] = r2_test
    results["final_test_r2s"] = final_test_r2s
    results["final_test_loss"] = loss
    return embeddings_stim, results


def test_time_adaptation(
    cfg,
    model,
    embeddings_rest,
    data_train,
    epochs=1000,
    data_test=None,
    ae_warm_start: bool = True,
    adapt_ae: bool = True,
    embeddings_stim=None,
    support_size: int | None = None,
    quiet: bool = False,
    coadapt_embeddings: bool = False,
) -> torch.Tensor:
    """
    Test-time adaptation for new sessions.

    Args:
        cfg: Config object
        model: Multisession model
        embeddings_rest: Rest embeddings for each session
        data_train: Training data for adaptation
        epochs: Number of adaptation steps
        data_test: Optional test data for evaluation
        ae_warm_start: If True, warm start AE with PCA initialization
        adapt_ae: If True, optimize the autoencoder weights
        embeddings_stim: If None, train new embeddings. Otherwise use provided and don't optimize.
        support_size: Override cfg.meta.training.support_size if provided.
        quiet: If True, suppress progress messages
        coadapt_embeddings: If True, optimize embeddings jointly with AE parameters

    Returns:
        embeddings_stim: Session embeddings (optimized or provided)
        results: Dict with test metrics
    """

    # Set model to eval mode; keep AE in train mode if adapting it
    model.eval(ae=not adapt_ae)

    device = model.device

    support_size = support_size or cfg.meta.training.support_size
    inner_steps = cfg.meta.training.inner_steps

    # We materialize the training data set under the presumption it is small and a single batch.
    # TODO we should probably enforce that somehow.
    _, data_train = utils.iter_loader(iter(data_train), data_train, device=device)

    # Apply outlier filtering to training data
    data_train, filter_stats = utils.filter_batch_outliers(data_train, model.norms, cfg)
    if cfg.training.use_outlier_filtering:
        print(
            f"TTA: Training data filtered: {filter_stats['total_kept']}/{filter_stats['total_samples']} trials kept"
        )

    # Split into support/query if support_size is provided
    data_for_adaptation, _ = split_support_query_sessions(data_train, support_size)
    print(f"TTA: Using {support_size} samples for adaptation (support set)")

    # AE warm start initialization using support data
    if ae_warm_start:
        print("TTA: Warm starting autoencoder...")
        with torch.no_grad():
            for session_id, data in data_for_adaptation.items():
                runway = data[0]  # (batch, time, channels)
                runway_normalized = model.norms({session_id: runway})

                # Get session-specific AE instance
                ae_instance = model.ae.instances.get(session_id)
                if ae_instance is not None:
                    ae_instance.pca_warm_start(
                        runway_normalized[session_id],
                        center="median",
                        whiten=False,
                    )
                    print(f"  Warm started AE for {session_id}")

    # Determine if we need to optimize embeddings or use hypernetwork
    optimize_embeddings = embeddings_stim is None
    use_hypernetwork = cfg.meta.use_hypernetwork if hasattr(cfg.meta, 'use_hypernetwork') else False
    support_contexts = None

    # Initialize embeddings as trainable parameters if co-adapting (MAML mode only)
    if optimize_embeddings and coadapt_embeddings and not use_hypernetwork:
        embed_dim_stim = model.model.bases.embed_dim_stim
        embeddings_stim = {}
        for session_id in embeddings_rest.keys():
            emb = torch.randn(embed_dim_stim, device=device) * 0.1
            emb.requires_grad = True
            embeddings_stim[session_id] = emb

    if optimize_embeddings and not adapt_ae and not coadapt_embeddings:
        # Only adapting meta-learning component (not AE)
        if use_hypernetwork:
            # Hypernetwork mode: encode support set to contexts
            print("TTA: Encoding support set with hypernetwork...")
            support_contexts = meta.hypernetwork_adapt(
                model,
                data_for_adaptation,
                embeddings_rest,
                cfg,
            )
            embeddings_stim = None  # Not used in hypernetwork mode
        else:
            # MAML mode: optimize embeddings via gradient descent
            print("TTA: Optimizing embeddings...")
            embeddings_stim, _ = meta.inner_update_stopgrad(
                model,
                data_for_adaptation,
            embeddings_rest,
            cfg,
            inner_steps=epochs,
            quiet=quiet,
        )

    # AE optimization (alone or jointly with embeddings)
    if adapt_ae:
        if coadapt_embeddings:
            print("TTA: Joint optimization of AE and embeddings...")
        else:
            print("TTA: Meta-learning style AE optimization...")

        # Outer loop: adapt AE parameters (and embeddings if co-adapting)
        # Inner loop: adapt embeddings (only if not co-adapting)
        # Both loops use data_for_adaptation (the tiny support set)

        # Set up AE optimizers
        ae_optims = []
        for session_id in data_for_adaptation.keys():
            ae_instance = model.ae.instances.get(session_id)
            if ae_instance is not None:
                ae_optim = torch.optim.AdamW(
                    ae_instance.parameters(),
                    lr=cfg.ae.training.optim.lr,
                    weight_decay=cfg.ae.training.optim.weight_decay,
                )
                ae_optims.append(ae_optim)

        # Set up embeddings optimizer if co-adapting
        embed_optim = None
        if coadapt_embeddings and optimize_embeddings:
            embed_params = [emb for emb in embeddings_stim.values()]
            embed_optim = torch.optim.AdamW(
                embed_params,
                lr=cfg.meta.training.optim.lr,
                weight_decay=cfg.meta.training.optim.weight_decay,
            )

        # Initialize variables for results
        ys = None
        yhat = None

        if coadapt_embeddings:
            print(f"  Running {epochs} joint optimization steps")
            print(
                f"  Using data_for_adaptation ({len(next(iter(data_for_adaptation.values()))[0])} samples)"
            )
        else:
            print(f"  Running {epochs} outer steps, each with {inner_steps} inner steps")
            print(
                f"  Both inner and outer loops use data_for_adaptation ({len(next(iter(data_for_adaptation.values()))[0])} samples)"
            )

        for outer_step in range(epochs):
            if coadapt_embeddings:
                # Joint optimization: update both AE and embeddings together
                # Process all sessions in one batched forward pass (like training)
                for opt in ae_optims:
                    opt.zero_grad()
                if embed_optim:
                    embed_optim.zero_grad()

                # Forward pass with all sessions at once
                yhat = model(
                    data_for_adaptation,
                    embeddings_rest=embeddings_rest,
                    embeddings_stim=embeddings_stim,
                )

                # Normalize y values for loss computation
                y_data = {sid: data[2] for sid, data in data_for_adaptation.items()}
                y_normalized = model.norms(y_data)

                # Compute combined loss across all sessions
                losses = {}
                for sid, y in y_normalized.items():
                    losses[sid] = nn.MSELoss()(yhat[sid], y)

                loss = sum(losses.values()) / len(data_for_adaptation)

                # Single backward pass
                loss.backward()

                # Update parameters
                for opt in ae_optims:
                    opt.step()
                if embed_optim:
                    embed_optim.step()

                if outer_step % 100 == 0 and not quiet:
                    print(f"  Joint step {outer_step}/{epochs}, loss: {loss.item():.6f}")

            else:
                # Inner loop: optimize embeddings on data_for_adaptation
                embeddings_stim_adapted = meta.inner_update_stopgrad(
                    model,
                    data_for_adaptation,
                    embeddings_rest,
                    cfg,
                    inner_steps=inner_steps,
                    quiet=True,
                )

                # Outer step: update AE on data_for_adaptation using adapted embeddings
                # Process all sessions in one batched forward pass (like training)
                for opt in ae_optims:
                    opt.zero_grad()

                # Forward pass with all sessions at once
                yhat = model(
                    data_for_adaptation,
                    embeddings_rest=embeddings_rest,
                    embeddings_stim=embeddings_stim_adapted,
                )

                # Normalize y values for loss computation
                y_data = {sid: data[2] for sid, data in data_for_adaptation.items()}
                y_normalized = model.norms(y_data)

                # Compute combined loss across all sessions
                losses = {}
                for sid, y in y_normalized.items():
                    losses[sid] = nn.MSELoss()(yhat[sid], y)

                loss = sum(losses.values()) / len(data_for_adaptation)

                # Single backward pass
                loss.backward()

                # Update AE parameters
                for opt in ae_optims:
                    opt.step()

                if outer_step % 100 == 0 and not quiet:
                    print(f"  Outer step {outer_step}/{epochs}, loss: {loss.item():.6f}")

        # Store final predictions for results (recompute with all sessions at once)
        if coadapt_embeddings or not optimize_embeddings:
            with torch.no_grad():
                yhat = model(
                    data_for_adaptation,
                    embeddings_rest=embeddings_rest,
                    embeddings_stim=embeddings_stim,
                )

                y_data = {sid: data[2] for sid, data in data_for_adaptation.items()}
                ys = model.norms(y_data)

        # After outer loop, do final inner optimization for embeddings to return (if not co-adapting)
        if not coadapt_embeddings:
            embeddings_stim, _ = meta.inner_update_stopgrad(
                model,
                data_for_adaptation,
                embeddings_rest,
                cfg,
                inner_steps=inner_steps,
                quiet=False,
            )

        if coadapt_embeddings:
            print(
                f"TTA: Joint optimization complete. Final loss: {loss.item():.6f}"
            )
        else:
            print(
                f"TTA: Meta-learning optimization complete."
            )
    else:
        ys = None
        yhat = None

    if data_test:
        with torch.no_grad():
            test_results = utils.evaluate_test_batches(
                model,
                data_test,
                embeddings_rest,
                embeddings_stim,
                model.norms,
                cfg,
                device,
                track_per_session_r2=True,
                support_contexts=support_contexts,
            )

            r2_test = test_results["r2"]
            loss = test_results["loss"]
            final_test_r2s = test_results["per_session_r2"]
            y_hat_test = test_results["y_hat"]
            test_batch = test_results["y_test"]

            if test_results["outlier_stats"] is not None:
                print(
                    f"TTA: Test data filtered: {test_results['outlier_stats']['total_kept']}/"
                    f"{test_results['outlier_stats']['total_samples']} "
                    f"({test_results['outlier_stats']['pct_kept']:.1f}%)"
                )
            print(f"TTA: Test results - Loss: {loss:.6f}, R2: {r2_test:.4f}")
    else:
        r2_test = None
        final_test_r2s = None
        loss = None
        y_hat_test = None
        test_batch = None

    results = {}
    results["final_test_r2"] = r2_test
    results["final_test_r2s"] = final_test_r2s
    results["final_test_loss"] = loss
    results["y"] = ys
    results["y_hat"] = yhat
    results["y_test"] = test_batch
    results["y_hat_test"] = y_hat_test
    return embeddings_stim, results


# Multisession batched data loading ----------------------------
def gather_session_ids(data_dir, num_held_out_sessions, held_in_session_ids=None):
    paths = [
        dd
        for dd in os.listdir(data_dir)
        if dd.startswith("Monkey") and os.path.isdir(os.path.join(data_dir, dd))
    ]
    held_in_session_ids = held_in_session_ids or random.sample(
        paths, len(paths) - num_held_out_sessions
    )
    held_out_session_ids = list(set(paths) - set(held_in_session_ids))

    return held_in_session_ids, held_out_session_ids


def load_stim_batched(
    runway=20,
    batch_size=1000,
    window_size=184,
    session_subdir="torchraw",
    data_dir="/var/data/opto-coproc/",
    held_in_session_ids=None,
    num_held_out_sessions=10,
    unpack_stiminds=True,
):
    held_in_session_ids, held_out_session_ids = gather_session_ids(
        data_dir, num_held_out_sessions, held_in_session_ids=held_in_session_ids
    )

    # We load to CPU initially for the purpose of streaming from disk and pinning,
    #  then we shunt to DEVICE later.
    d = dataset.load_data_some_sessions(
        runway=runway,
        paths=held_in_session_ids,
        subdir=data_dir,
        session_subdir=session_subdir,
        batch_size=batch_size,
        window_size=window_size,
        unpack_stiminds=unpack_stiminds,
        in_memory=True,
        device="cpu",
    )

    return d, held_out_session_ids


def load_rest_embeddings(
    session_ids, in_dir="/var/data/opto-coproc/", in_subdir="embedding_rest", device=None
):
    """
    See meta.cache_rest_embeds.
    """
    embeddings_rest = {}
    for session_id in session_ids:
        path = os.path.join(in_dir, session_id, in_subdir, "er.torch")
        er = torch.load(path).to(device)
        embeddings_rest[session_id] = er
    return embeddings_rest
