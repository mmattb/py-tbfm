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
from . import film
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
    latent_dim = cfg["latent_dim"]

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
            use_lora=shared_ae,
            latent_dim=latent_dim,
            shared=shared_ae,
            warm_start=True,
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


def get_optims(cfg, model_ms: TBFMMultisession):
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
    film_optims = []

    for sid, optims in optims_model.items():
        optim_bw, optim_bg, optim_film = optims

        bw_optims.append(optim_bw)
        bw_schedulers.append(LambdaLR(optim_bw, lr_lambda=warmup_cos(1500, cos=False)))

        bg_optims.append(optim_bg)
        bg_schedulers.append(LambdaLR(optim_bg, lr_lambda=warmup_cos(5000)))

        if optim_film is not None:
            film_optims.append(optim_film)

    # Create named optimizer groups
    groups = {
        "norm": {"optimizers": optims_norms, "schedulers": []},
        "ae": {"optimizers": optims_aes, "schedulers": []},
        "bw": {"optimizers": bw_optims, "schedulers": bw_schedulers},
        "bg": {"optimizers": bg_optims, "schedulers": bg_schedulers},
        "film": {"optimizers": film_optims, "schedulers": []},
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
    data_test=None,
    inner_steps: int | None = None,
    epochs: int = 10000,
    test_interval: int | None = None,
    support_size: int = 300,
    embed_stim_lr: int | None = None,
    embed_stim_weight_decay: float = None,
    device: str = "cuda",
    grad_clip: float | None = None,
    alternating_updates: bool = True,  # Use alternating basis weight / basis gen updates
    bw_steps_per_bg_step: (
        int | None
    ) = None,  # How many basis weight updates per basis update
    model_save_path: str | None = None,
):
    """
    One epoch over sessions with:
      - inner loop on support to adapt stim_embed (per-episode latent)
      - outer update on query for shared params
      - slow AE updates every step (small lr)
      - optional alternating updates: update basis weights more frequently than basis generator

    Alternating updates rationale:
      - Basis weights have large gradients and adapt quickly
      - Basis generator (Bases) changes more slowly
      - By updating basis weights N times per basis generator update, we let the
        weighting layer stabilize to the current bases before changing the bases

    Notes:
      • To add EMA for AE: keep a shadow copy of AE params and update with EMA after each optimizer.step().
    """
    # cfg overrides
    test_interval = test_interval or cfg.training.test_interval
    embed_stim_lr = embed_stim_lr or cfg.film.training.optim.lr
    embed_stim_weight_decay = (
        embed_stim_weight_decay or cfg.film.training.optim.weight_decay
    )
    inner_steps = inner_steps or cfg.film.training.inner_steps
    use_film = cfg.tbfm.module.use_film_bases
    bw_steps_per_bg_step = bw_steps_per_bg_step or cfg.training.bw_steps_per_bg_step
    grad_clip = grad_clip or cfg.training.grad_clip or 10.0
    epochs = epochs or cfg.training.epochs
    support_size = support_size or cfg.film.training.support_size
    ae_freeze_epoch = cfg.training.get("ae_freeze_epoch", None)
    device = model.device

    embeddings_stim = None  # default
    iter_train = iter(data_train)

    train_losses = []
    train_r2s = []
    test_losses = []
    test_r2s = []
    min_test_r2 = 1e99
    for eidx in range(epochs):
        model.train()
        iter_train, _data_train = utils.iter_loader(
            iter_train, data_train, device=device
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
        if use_film:
            model.eval()
            embeddings_stim = film.inner_update_stopgrad(
                model,
                support,
                embeddings_rest,
                inner_steps=inner_steps,
                lr=embed_stim_lr,
                weight_decay=embed_stim_weight_decay,
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

        train_losses.append((eidx, loss.item()))

        loss.backward()

        if grad_clip is not None:
            model_optims.clip_grad(value=grad_clip)

        # if eidx == 3000:
        #    utils.log_grad_norms(model.model.instances["MonkeyG_20150925_Session2_S1"])
        #    utils.log_grad_norms(model.ae.instances["MonkeyG_20150925_Session2_S1"])

        # Alternating updates: update basis weights more frequently than basis generator
        # Also: freeze AE after specified epoch to prevent late-stage overfitting
        skip_groups = []

        if ae_freeze_epoch is not None and eidx >= ae_freeze_epoch:
            skip_groups.append("ae")

        if alternating_updates:
            # Every basis_weight_steps_per_basis iterations, update both
            # Otherwise, only update basis weights
            update_basis_gen = (eidx % bw_steps_per_bg_step) == 0 or eidx < 200

            if update_basis_gen:
                # Update everything (basis weights, basis gen, film, ae, norm) minus frozen groups
                model_optims.step(skip=skip_groups)
            else:
                # Update only basis weights (skip basis gen and film) minus frozen groups
                model_optims.step(skip=["bg", "film"] + skip_groups)
        else:
            model_optims.step(skip=skip_groups)

        if data_test is not None and (eidx % test_interval) == 0:
            train_r2s.append((eidx, sum(r2_trains) / len(y_query)))

            model_optims.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.eval()
                loss_outer = 0
                r2_outer = 0
                test_batch_count = 0
                for test_batch in data_test:
                    test_batch = utils.move_batch(test_batch, device=device)

                    tb_norm = {}
                    for session_id, d in test_batch.items():
                        y_test = model.norms.instances[session_id](d[2])
                        new_d = (d[0], d[1], y_test)
                        tb_norm[session_id] = new_d
                    test_batch = tb_norm

                    y_hat_test = model(
                        test_batch,
                        embeddings_rest=embeddings_rest,
                        embeddings_stim=embeddings_stim,
                    )

                    loss = 0
                    r2_test = 0
                    for session_id, d in test_batch.items():
                        y_test = d[2]
                        loss += nn.MSELoss()(y_hat_test[session_id], y_test)
                        r2_test += r2_score(
                            y_hat_test[session_id].permute(0, 2, 1).flatten(end_dim=1),
                            y_test.permute(0, 2, 1).flatten(end_dim=1),
                        )
                    loss /= len(test_batch)
                    r2_test /= len(test_batch)

                    r2_outer += r2_test.item()
                    loss_outer += loss.item()
                    test_batch_count += 1

                r2_test = r2_outer / test_batch_count
                loss = loss_outer / test_batch_count
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

    if use_film:
        iter_train, _data_train = utils.iter_loader(
            iter_train, data_train, device=device
        )
        # split into support/query
        support, _ = split_support_query_sessions(
            _data_train,
            support_size=support_size,
        )

        model_optims.zero_grad(set_to_none=True)
        embeddings_stim = film.inner_update_stopgrad(
            model,
            support,
            embeddings_rest,
            inner_steps=(3 * inner_steps),
            lr=embed_stim_lr,
            weight_decay=embed_stim_weight_decay,
        )

    model_optims.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.eval()
        loss_outer = 0
        r2_outer = 0
        test_batch_count = 0
        final_test_r2s = {session_id: 0 for session_id in test_batch.keys()}
        for test_batch in data_test:
            test_batch = utils.move_batch(test_batch, device=device)
            tb_norm = {}
            for session_id, d in test_batch.items():
                y_test = model.norms.instances[session_id](d[2])
                new_d = (d[0], d[1], y_test)
                tb_norm[session_id] = new_d
            test_batch = tb_norm

            y_hat_test = model(
                test_batch,
                embeddings_rest=embeddings_rest,
                embeddings_stim=embeddings_stim,
            )

            loss = 0
            r2_test = 0
            for session_id, d in test_batch.items():
                y_test = d[2]
                loss += nn.MSELoss()(y_hat_test[session_id], y_test)
                _r2_test = r2_score(
                    y_hat_test[session_id].permute(0, 2, 1).flatten(end_dim=1),
                    y_test.permute(0, 2, 1).flatten(end_dim=1),
                )
                r2_test += _r2_test
                final_test_r2s[session_id] += _r2_test.item()
            loss /= len(test_batch)
            r2_test /= len(test_batch)

            r2_outer += r2_test.item()
            loss_outer += loss.item()
            test_batch_count += 1

        r2_test = r2_outer / test_batch_count
        loss = loss_outer / test_batch_count
        final_test_r2s = {
            session_id: r2 / test_batch_count
            for session_id, r2 in final_test_r2s.items()
        }

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
    lr=None,
    weight_decay=None,
    grad_clip=None,
) -> torch.Tensor:

    model.eval()

    lr = lr or cfg.film.training.optim.lr
    weight_decay = weight_decay or cfg.film.training.optim.weight_decay
    grad_clip = grad_clip or cfg.training.grad_clip or 10.0
    device = model.device

    # We materialize the training data set under the presumption it is small and a single batch.
    # TODO we should probably enforce that somehow.
    _, data_train = utils.iter_loader(iter(data_train), data_train, device=device)

    # TODO: need to do the AE warm starting here.

    embeddings_stim, _ = film.inner_update_stopgrad(
        model,
        data_train,
        embeddings_rest,
        inner_steps=epochs,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        quiet=False,
    )

    if data_test:
        with torch.no_grad():
            loss_outer = 0
            r2_outer = 0
            test_batch_count = 0
            final_test_r2s = {session_id: 0 for session_id in data_train.keys()}
            for test_batch in data_test:
                test_batch = utils.move_batch(test_batch, device=device)
                tb_norm = {}
                for session_id, d in test_batch.items():
                    y_test = model.norms.instances[session_id](d[2])
                    new_d = (d[0], d[1], y_test)
                    tb_norm[session_id] = new_d
                test_batch = tb_norm

                y_hat_test = model(
                    test_batch,
                    embeddings_rest=embeddings_rest,
                    embeddings_stim=embeddings_stim,
                )

                loss = 0
                r2_test = 0
                for session_id, d in test_batch.items():
                    y_test = d[2]
                    loss += nn.MSELoss()(y_hat_test[session_id], y_test)
                    _r2_test = r2_score(
                        y_hat_test[session_id].permute(0, 2, 1).flatten(end_dim=1),
                        y_test.permute(0, 2, 1).flatten(end_dim=1),
                    )
                    r2_test += _r2_test
                    final_test_r2s[session_id] += _r2_test.item()
                loss /= len(test_batch)
                r2_test /= len(test_batch)

                r2_outer += r2_test.item()
                loss_outer += loss.item()
                test_batch_count += 1

            r2_test = r2_outer / test_batch_count
            loss = loss_outer / test_batch_count
            final_test_r2s = {
                session_id: r2 / test_batch_count
                for session_id, r2 in final_test_r2s.items()
            }
    else:
        r2_test = None
        final_test_r2s = None
        loss = None

    results = {}
    results["final_test_r2"] = r2_test
    results["final_test_r2s"] = final_test_r2s
    results["final_test_loss"] = loss
    results["y_hat_test"] = y_hat_test
    results["y_test"] = test_batch
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
    data_dir="/home/mmattb/Projects/opto-coproc/data",
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
    session_ids, in_dir="data", in_subdir="embedding_rest", device=None
):
    """
    See film.cache_rest_embeds.
    """
    embeddings_rest = {}
    for session_id in session_ids:
        path = os.path.join(in_dir, session_id, in_subdir, "er.torch")
        er = torch.load(path).to(device)
        embeddings_rest[session_id] = er
    return embeddings_rest


# TODO: inner loop regularization
# TODO: AE regularization including orthogonality
