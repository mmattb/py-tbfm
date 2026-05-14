import math
import random
import os
import sys

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Tuple, List

from . import ae
from . import dataset
from . import meta
from . import normalizers
from . import tbfm
from . import utils
from .utils import r2_score

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
    latent_dim = cfg.latent_dim

    with torch.no_grad():
        # Normalizers ------
        if not quiet:
            print("Building and fitting normalizers...")
        norms = normalizers.from_cfg(cfg, session_data, device=device)

        # Autoencoders ------
        if not quiet:
            print("Building and warm starting AEs...")
        aes = ae.from_cfg_and_data(
            cfg,
            session_data,
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


def save_model(model, path):
    """Save full model to a directory (split format: tbfm.torch, ae.torch, norms.torch)."""
    os.makedirs(path, exist_ok=True)
    instances = set(model.model.instances.values())
    if len(instances) != 1:
        raise NotImplementedError("Only single shared TBFM supported for save")
    model_tbfm = next(iter(instances))
    torch.save(model_tbfm.state_dict(), os.path.join(path, "tbfm.torch"))
    ae_states = {sid: inst.state_dict() for sid, inst in model.ae.instances.items()}
    torch.save(ae_states, os.path.join(path, "ae.torch"))
    norm_states = {
        sid: inst.state_dict() for sid, inst in model.norms.instances.items()
    }
    torch.save(norm_states, os.path.join(path, "norms.torch"))


def get_optims(cfg, model_ms: TBFMMultisession, embeddings_stim=None):
    optims_norms = tuple()
    if cfg.normalizers.training.coadapt:
        raise NotImplementedError("No normalizer adaptation yet")

    if cfg.ae.training.coadapt:
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

    # Optional optimizer for trainable per-session stim embeddings (used when
    # cfg.meta.training.coadapt=true, i.e. outer-loop co-adaptation instead of
    # MAML inner-loop adaptation).
    embed_optims = []
    if embeddings_stim is not None:
        embed_lr = cfg.meta.training.optim.lr
        embed_wd = cfg.meta.training.optim.weight_decay
        embed_optims.append(
            torch.optim.AdamW(
                list(embeddings_stim.values()), lr=embed_lr, weight_decay=embed_wd
            )
        )

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
    random_sample: bool = True,
):
    support = {}
    query = {}
    for session_id, d in data_train.items():
        if random_sample:
            n_samples = len(d[0])
            indices = torch.randperm(n_samples)
            support_indices = indices[:support_size]
            query_indices = indices[support_size:]
            support[session_id] = tuple(dd[support_indices] for dd in d)
            query[session_id] = tuple(dd[query_indices] for dd in d)
        else:
            support[session_id] = tuple(dd[:support_size] for dd in d)
            query[session_id] = tuple(dd[support_size:] for dd in d)

    return support, query


def train_from_cfg(
    cfg,
    model: nn.Module,
    data_train,  # yields per-session batches (stiminds, runways, y)
    model_optims,
    embeddings_rest,
    embeddings_stim=None,  # Pre-initialized trainable stim embeddings for co-adaptation
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
    model_save_path: str | None = None,
    random_sample_support: bool = True,  # Randomly sample support set instead of sequential
):
    """
    One epoch over sessions with:
      - inner loop on support to adapt stim_embed (per-episode latent)  -- MAML mode
        OR
      - outer-loop co-adaptation of per-session embeddings (cfg.meta.training.coadapt)
      - outer update on query for shared params
      - slow AE updates every step (small lr)
      - optional alternating updates: update basis weights more frequently than basis generator

    Alternating updates rationale:
      - Basis weights have large gradients and adapt quickly
      - Basis generator (Bases) changes more slowly
      - By updating basis weights N times per basis generator update, we let the
        weighting layer stabilize to the current bases before changing the bases
      - When co-adapting embeddings, alternate between embedding and other-param updates

    Notes:
      • To add EMA for AE: keep a shadow copy of AE params and update with EMA after each optimizer.step().
    """
    # cfg overrides
    test_interval = test_interval or cfg.training.test_interval
    use_meta = cfg.tbfm.module.use_meta_learning
    coadapt_embeddings = (
        bool(cfg.meta.training.get("coadapt", False)) if use_meta else False
    )
    embed_steps_per_other_step = int(
        cfg.meta.training.get("embed_steps_per_other_step", 1)
    )
    bw_steps_per_bg_step = bw_steps_per_bg_step or cfg.training.bw_steps_per_bg_step
    grad_clip = grad_clip or cfg.training.grad_clip or 10.0
    epochs = epochs or cfg.training.epochs
    support_size = support_size or cfg.meta.training.support_size
    ae_freeze_epoch = cfg.ae.training.ae_freeze_epoch
    lambda_ae_recon = cfg.ae.training.lambda_ae_recon
    device = model.device

    if coadapt_embeddings and embeddings_stim is None:
        # Initialize trainable stim embeddings if user requested co-adaptation
        # but didn't pre-create them (e.g. for warm-start). These should be
        # added to the optimizer via get_optims(..., embeddings_stim=...).
        raise ValueError(
            "cfg.meta.training.coadapt=true requires passing trainable "
            "embeddings_stim into train_from_cfg AND registering them with "
            "get_optims(..., embeddings_stim=...)."
        )
    iter_train = iter(data_train)

    train_losses = []
    train_r2s = []
    test_losses = []
    test_r2s = []
    max_test_r2 = -1e99

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
            random_sample=random_sample_support,
        )

        with torch.no_grad():
            y_query = {sid: d[2] for sid, d in query.items()}
            y_query = model.norms(y_query)

        # ----- inner adaptation on support -----
        # If coadapt_embeddings is true, skip MAML inner loop and use the
        # outer-optimized embeddings_stim directly.
        if use_meta and not coadapt_embeddings:
            model.eval()
            embeddings_stim = meta.inner_update_stopgrad(
                model,
                support,
                embeddings_rest,
                cfg,
            )
            model.train()

        model_optims.zero_grad(set_to_none=True)

        yhat_query = model(
            query, embeddings_rest=embeddings_rest, embeddings_stim=embeddings_stim
        )

        losses = {}
        r2_trains = []
        for sid, y in y_query.items():
            _loss = nn.MSELoss()(yhat_query[sid], y)
            losses[sid] = _loss

            r2_train = r2_score(
                yhat_query[sid].permute(0, 2, 1).flatten(end_dim=1),
                y.permute(0, 2, 1).flatten(end_dim=1),
            )
            r2_trains.append(r2_train.item())

        cur_loss = sum(losses.values()) / len(y_query)
        train_losses.append((eidx, cur_loss.item()))

        # Add AE reconstruction loss (optional)
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

        if ae_freeze_epoch is not None and eidx >= ae_freeze_epoch:
            skip_groups.append("ae")

        if alternating_updates:
            # Every basis_weight_steps_per_basis iterations, update both
            # Otherwise, only update basis weights
            update_basis_gen = (eidx % bw_steps_per_bg_step) == 0 or eidx < 200

            if coadapt_embeddings:
                # Also gate the 'embed' group so the trainable session embeddings
                # don't fight the basis weighting on every step.
                update_embeddings = (
                    eidx % embed_steps_per_other_step
                ) == 0 or eidx < 200

                if update_embeddings and update_basis_gen:
                    model_optims.step(skip=skip_groups)
                elif update_embeddings:
                    model_optims.step(skip=["bg", "meta"] + skip_groups)
                elif update_basis_gen:
                    model_optims.step(skip=["embed"] + skip_groups)
                else:
                    model_optims.step(skip=["bg", "meta", "embed"] + skip_groups)
            elif update_basis_gen:
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
                )

                loss = test_results["loss"]
                r2_test = test_results["r2"]
                test_losses.append((eidx, loss))
                test_r2s.append((eidx, r2_test))
                print(
                    "----", eidx, train_losses[-1][-1], loss, train_r2s[-1][-1], r2_test
                )

                if r2_test > max_test_r2:
                    max_test_r2 = r2_test
                    if model_save_path:
                        save_model(model, model_save_path)
                        torch.save(
                            {
                                "epoch": eidx,
                                "train_loss": train_losses[-1][-1],
                                "train_r2": train_r2s[-1][-1],
                                "test_loss": loss,
                                "test_r2": r2_test,
                            },
                            os.path.join(model_save_path, "best_metrics.torch"),
                        )

    # ----- (optional) EMA of AE params -----
    # for p, p_ema in zip(model.ae_parameters(), ae_ema_params):
    #     p_ema.data.mul_(1 - ema_alpha).add_(p.data, alpha=ema_alpha)

    use_meta = cfg.tbfm.module.use_meta_learning
    if use_meta and not coadapt_embeddings:
        iter_train, _data_train = utils.iter_loader(
            iter_train, data_train, device=device
        )
        # split into support/query
        support, _ = split_support_query_sessions(
            _data_train,
            support_size=support_size,
            random_sample=random_sample_support,
        )

        model_optims.zero_grad(set_to_none=True)
        embeddings_stim = meta.inner_update_stopgrad(
            model,
            support,
            embeddings_rest,
            cfg,
            inner_steps=(3 * cfg.meta.training.inner_steps),
        )

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

        if r2_test > max_test_r2:
            max_test_r2 = r2_test
            if model_save_path:
                save_model(model, model_save_path)
                torch.save(
                    {
                        "epoch": epochs - 1,
                        "train_loss": train_losses[-1][-1] if train_losses else None,
                        "train_r2": train_r2s[-1][-1] if train_r2s else None,
                        "test_loss": loss,
                        "test_r2": r2_test,
                    },
                    os.path.join(model_save_path, "best_metrics.torch"),
                )

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

    # Refit normalizers from support set (not from the full training set)
    print("TTA: Refitting normalizers from support set...")
    with torch.no_grad():
        for session_id, d in data_for_adaptation.items():
            x = torch.cat((d[0], d[2]), dim=1)  # runway + targets
            model.norms.instances[session_id].fit(x)

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

    # Determine if we need to optimize embeddings
    optimize_embeddings = embeddings_stim is None

    if optimize_embeddings and not adapt_ae:
        # Only optimizing embeddings (not AE)
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
    # AE optimization using meta-learning approach
    if adapt_ae:
        print("TTA: Meta-learning style AE optimization...")

        # Outer loop: adapt AE parameters
        # Inner loop: adapt embeddings
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

        print(f"  Running {epochs} outer steps, each with {inner_steps} inner steps")
        print(
            f"  Both inner and outer loops use data_for_adaptation ({len(next(iter(data_for_adaptation.values()))[0])} samples)"
        )

        for outer_step in range(epochs):
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
            for opt in ae_optims:
                opt.zero_grad()

            # Forward on data_for_adaptation with adapted embeddings
            yhat = model(
                data_for_adaptation,
                embeddings_rest=embeddings_rest,
                embeddings_stim=embeddings_stim_adapted,
            )

            # Normalize y values for loss computation (similar to train_from_cfg line 260)
            # Note: inner_update_stopgrad already normalizes internally, so we only normalize here
            y_data = {sid: d[2] for sid, d in data_for_adaptation.items()}
            y_normalized = model.norms(y_data)

            # Compute loss on data_for_adaptation
            loss = 0
            ys = {}
            for session_id in data_for_adaptation.keys():
                y = y_normalized[session_id]
                loss += nn.MSELoss()(yhat[session_id], y)
                ys[session_id] = y.detach()
            loss = loss / len(data_for_adaptation)

            # Backward and update AE
            loss.backward()
            for opt in ae_optims:
                opt.step()

            if outer_step % 1000 == 0 and not quiet:
                print(f"  Outer step {outer_step}/{epochs}, loss: {loss.item():.6f}")

        # After outer loop, do final inner optimization for embeddings to return
        embeddings_stim, _ = meta.inner_update_stopgrad(
            model,
            data_for_adaptation,
            embeddings_rest,
            cfg,
            inner_steps=inner_steps,
            quiet=False,
        )

        print(
            f"TTA: Meta-learning optimization complete. Final loss: {loss.item():.6f}"
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
    data_dir=None,
    held_in_session_ids=None,
    num_held_out_sessions=10,
    unpack_stiminds=True,
):

    if not data_dir:
        raise ValueError("Must supply data_dir")

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
    session_ids, in_dir="data", in_subdir="embedding_rest", device=None
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
