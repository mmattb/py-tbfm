# Okay; this is a bit complicated...
# In general: some args will come in dictionaries {session_id, arg,...}, others will be bound by a closure.
import random
import os
import sys

import torch
from torch import nn
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
    device=None,
):
    latent_dim = cfg["latent_dim"]

    with torch.no_grad():
        # Normalizers ------
        print("Building and fitting normalizers...")
        norms = normalizers.from_cfg(cfg, session_data, device=device)

        # Autoencoders ------
        print("Building and warm starting AEs...")
        aes = ae.from_cfg_and_data(
            cfg,
            session_data,
            use_bias=True,
            use_lora=shared_ae,
            latent_dim=latent_dim,
            shared=shared_ae,
            warm_start=True,
            device=device,
        )

        # TBFM ------
        print("Building TBFM...")
        _tbfm = tbfm.from_cfg(
            cfg,
            tuple(session_data.keys()),
            shared=cfg.tbfm.sharing.shared,
            device=device,
        )

        # Cleared for takeoff ------
        print("BOOM! Dino DNA!")
    return TBFMMultisession(norms, aes, _tbfm, device=device)


def get_optims(cfg, model_ms: TBFMMultisession):
    optims_norms = tuple()
    if cfg.normalizers.training.coadapt:
        raise NotImplementedError("No normalizer adaptation yet")

    if cfg.ae.training.coadapt:
        optims_aes = set(model_ms.ae.get_optim(**cfg.ae.training.optim).values())
    else:
        optims_aes = tuple()

    # Will have only 1 elem if tbfm is shared.
    optims_models = set(model_ms.model.get_optim(**cfg.tbfm.training.optim).values())

    return utils.OptimCollection((optims_norms, optims_aes, optims_models))


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
    inner_mode: str = "stopgrad",  # "stopgrad" | "maml"
    inner_steps: int | None = None,
    epochs: int = 10000,
    test_interval: int | None = None,
    support_size: int = 100,
    inner_lr: float = 5e-2,
    ae_lr: float | None = None,
    embed_stim_lr: int | None = None,
    embed_stim_weight_decay: float = None,
    regularizer_fn=None,  # callable(model, rest_embed, stim_embed) -> scalar penalty
    device: str = "cuda",
    grad_clip: float = 1.0,
):
    """
    One epoch over sessions with:
      - inner loop on support to adapt stim_embed (per-episode latent)
      - outer update on query for shared params
      - slow AE updates every step (small lr)

    Notes:
      • To add EMA for AE: keep a shadow copy of AE params and update with EMA after each optimizer.step().
    """
    # cfg overrides
    test_interval = test_interval or cfg.training.test_interval
    embed_stim_lr = embed_stim_lr or cfg.film.training.optim.lr
    embed_stim_weight_decay = (
        embed_stim_weight_decay or cfg.film.training.optim.weight_decay
    )
    embed_dim_rest = cfg.tbfm.module.embed_dim_rest
    inner_steps = inner_steps or cfg.film.training.inner_steps
    use_film = cfg.tbfm.module.use_film_bases
    device = model.device

    if inner_mode not in {"stopgrad", "maml"}:
        raise ValueError("inner_mode must be 'stopgrad' or 'maml'")

    embeddings_stim = None  # default
    iter_train = iter(data_train)
    iter_test = iter(data_test)

    train_losses = []
    test_losses = []
    test_r2s = []
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

        # ----- inner adaptation on support -----
        if use_film:
            if inner_mode == "stopgrad":
                embeddings_stim = film.inner_update_stopgrad(
                    model,
                    support,
                    embeddings_rest,
                    inner_steps=inner_steps,
                    lr=embed_stim_lr,
                    weight_decay=embed_stim_weight_decay,
                    # regularizer_fn=regularizer_fn,
                )
            else:  # "maml"
                raise NotImplementedError()
                # stim_embed_query = film.inner_update_maml(
                #    model=model,
                #    loss_fn=loss_fn,
                #    stiminds_support=parts["stiminds_support"],
                #    rest_embed_support=parts["rest_embed_support"],
                #    inputs_support=parts["inputs_support"],
                #    targets_support=parts["targets_support"],
                #    stim_dim=stim_dim,
                #    inner_steps=inner_steps,
                #    inner_lr=inner_lr,
                #    regularizer_fn=regularizer_fn,
                # )

        model_optims.zero_grad(set_to_none=True)

        yhat_query = model(
            query, embeddings_rest=embeddings_rest, embeddings_stim=embeddings_stim
        )

        losses = {}
        for sid, y in y_query.items():
            loss = nn.MSELoss()(yhat_query[sid], y)
            losses[sid] = loss
        loss = sum(losses.values()) / len(y_query)
        train_losses.append(loss.item())

        tbfm_regs = model.model.get_weighting_reg()
        loss += cfg.tbfm.training.lambda_fro * sum(tbfm_regs.values()) / len(y_query)

        # TODO: other regularization etc. here down
        # if regularizer_fn is not None:
        #    loss += loss + regularizer_fn(
        #        model, parts["rest_embed_query"], stim_embed_query
        #    )

        loss.backward()

        if grad_clip is not None:
            model_optims.clip_grad(value=grad_clip)

        model_optims.step()

        if data_test is not None and (eidx % test_interval) == 0:
            model_optims.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.eval()
                loss_outer = 0
                r2_outer = 0
                test_batch_count = 0
                for test_batch in data_test:
                    test_batch = utils.move_batch(test_batch, device=device)
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
                print("----", eidx, loss, r2_test)

    # ----- (optional) EMA of AE params -----
    # for p, p_ema in zip(model.ae_parameters(), ae_ema_params):
    #     p_ema.data.mul_(1 - ema_alpha).add_(p.data, alpha=ema_alpha)

    # TODO: full train set
    if inner_mode == "stopgrad":
        embeddings_stim = film.inner_update_stopgrad(
            model,
            _data_train,
            embeddings_rest,
            inner_steps=1000,
            lr=embed_stim_lr,
            weight_decay=embed_stim_weight_decay,
            # regularizer_fn=regularizer_fn,
        )
    else:
        raise NotImplementedError()

    model_optims.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.eval()
        loss_outer = 0
        r2_outer = 0
        test_batch_count = 0
        final_test_r2s = {session_id: 0 for session_id in test_batch.keys()}
        for test_batch in data_test:
            test_batch = utils.move_batch(test_batch, device=device)
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
                final_test_r2s[session_id] += _r2_test
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

    print("Final:", loss, r2_test)

    results = {}
    results["train_losses"] = train_losses
    results["test_losses"] = test_losses
    results["test_r2s"] = test_r2s
    results["y_hat"] = yhat_query
    results["y"] = y_query
    results["y_hat_test"] = y_hat_test
    results["y_test"] = test_batch
    results["final_test_r2"] = r2_test
    results["final_test_r2s"] = final_test_r2s
    results["final_test_loss"] = loss
    return embeddings_stim, results


# Multisession batched data loading ----------------------------
def load_stim_batched(
    runway=20,
    batch_size=1000,
    window_size=184,
    session_subdir="torchraw",
    data_dir="/home/mmattb/Projects/opto-coproc/data",
    held_in_session_ids=None,
    num_held_out_sessions=10,
):
    paths = [
        dd
        for dd in os.listdir(data_dir)
        if dd.startswith("Monkey") and os.path.isdir(os.path.join(data_dir, dd))
    ]
    held_in_session_ids = held_in_session_ids or random.sample(
        paths, len(paths) - num_held_out_sessions
    )
    held_out_session_ids = list(set(paths) - set(held_in_session_ids))

    # We load to CPU initially for the purpose of streaming from disk and pinning,
    #  then we shunt to DEVICE later.
    d = dataset.load_data_some_sessions(
        runway=runway,
        paths=held_in_session_ids,
        subdir=data_dir,
        session_subdir=session_subdir,
        batch_size=batch_size,
        window_size=window_size,
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
