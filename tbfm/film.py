import os

import torch
from torch import nn
from typing import Dict, Tuple, List

from . import acf
from . import utils

from ._multisession_module import TBFMMultisession


def dispatch(data, func):
    d_out = {}
    for sid, d in data.items():
        d_out[sid] = func(d)
    return d_out


# TTA embedding ------------------------------------------------------


def embed_rest(rest_data):
    """
    rest_data: (batch_size, time, ch)
    """
    aacfs = []  # Area under absolute value of ACF
    t = torch.arange(rest_data.shape[1])
    rest_data_cpu = rest_data.cpu()
    for cidx in range(rest_data.shape[2]):
        ch_acf = acf.get_acf(rest_data_cpu, cidx)
        # Normalized A-ACF for 100 lags
        ch_aacf = acf.calc_auacf(t, ch_acf.acf, startend=(0, 100))
        aacfs.append(ch_aacf.item())

    aacfs.sort()

    # Cool. Now we return the 0.1/0.5/0.9 IQR
    dec_idx = len(aacfs) // 10
    med_idx = len(aacfs) // 2
    non_idx = len(aacfs) - dec_idx

    vals = [aacfs[idx] for idx in (dec_idx, med_idx, non_idx)]

    with torch.no_grad():
        return torch.tensor(vals).to(rest_data.device)


def embed_rest_sessions(rest_datas):
    return dispatch(rest_datas, embed_rest)


def embed_stim_residual_sessions(
    data,
    embeddings_rest,
    model_ms: TBFMMultisession,
    model_optims,
    attempt_count=5,
    epochs=1000,
    lr=1e-2,
):
    """
    Return estimated optimal stim embeddings, given a fixed rest embedding set and
    model.  This is not the interactive training loop for the whole model; it's for
    TTA where the model is frozen except session-specific bits.
    data: (batch_size, time, ch)
    """
    if (
        model_ms.model.bases.embed_dim_rest is None
        or model_ms.model.bases.embed_dim_stim is None
    ):
        raise ValueError("Cannot embed stim without embed_dim_stim set")

    embed_dim_stim = model_ms.model.bases.embed_dim_stim

    with torch.no_grad():
        embeddings_stim = {
            sid: nn.Parameter(torch.zeros(embed_dim_stim).to(model_ms.device))
            for sid in data.keys()
        }
        _optims = [
            torch.optim.AdamW((embeddings_stim[sid],), lr=lr) for sid in data.keys()
        ]
        optims = utils.OptimCollection(_optims)

        ys = {sid: d[2] for sid, d in data.items()}
        # {sid: ([loss,], [embedding,])}
        embeds = {sid: ([], []) for sid in data.keys()}

    model_ms.eval()
    for _ in range(attempt_count):
        optims.zero_grad()
        with torch.no_grad():
            for sid, embed in embeddings_stim.items():
                embed[:] = torch.randn(embed_dim_stim)
        model_optims.zero_grad()

        for _ in range(epochs):
            yhats = model_ms(
                data, embeddings_rest=embeddings_rest, embeddings_stim=embeddings_stim
            )

            losses = {}
            for sid, y in ys.items():
                loss = nn.MSELoss()(yhats[sid], y)
                losses[sid] = loss
            sum(losses.values()).backward()
            optims.step()

        for sid, loss in losses.items():
            embed_stim = embeddings_stim[sid]
            idx = len(embeds[sid][0])
            embeds[sid][0].append((loss.item(), idx))
            embeds[sid][1].append(embed_stim.clone())

    embeddings_out = {}
    for sid, (losses, candidates) in embeds.items():
        min_idx = min(losses)[1]  # idx of lowest loss embedding
        embeddings_out[sid] = candidates[
            min_idx
        ]  # i.e. embed which induced the lowest loss
    return embeddings_out


# Training ------------------------------------------------------


def inner_update_stopgrad(
    model: nn.Module,
    data_support,
    embeddings_rest,
    inner_steps: int = 100,
    lr: float = 5e-2,
    weight_decay: float = 1e-3,
) -> torch.Tensor:
    """
    Optimize a per-episode stim_embed on SUPPORT ONLY.
    Does NOT backprop through the inner loop (two-backprop variant).
    Returns a detached stim_embed to use on the QUERY pass.
    """
    embed_dim_stim = model.model.bases.embed_dim_stim
    # Split out y values for loss fn

    with torch.no_grad():
        ys = {sid: d[2] for sid, d in data_support.items()}
    device = ys[list(ys.keys())[0]].device

    embeddings_stim = {
        session_id: torch.zeros(embed_dim_stim, device=device, requires_grad=True)
        for session_id in data_support.keys()
    }
    optimizer_inner = utils.OptimCollection(
        [
            torch.optim.AdamW((se,), lr=lr, weight_decay=weight_decay)
            for se in embeddings_stim.values()
        ]
    )

    for _ in range(inner_steps):
        optimizer_inner.zero_grad()

        preds = model(
            data_support,
            embeddings_rest=embeddings_rest,
            embeddings_stim=embeddings_stim,
        )
        loss = 0
        for session_id, y in ys.items():
            loss += nn.MSELoss()(preds[session_id], y)
        loss /= len(preds)

        # if regularizer_fn is not None:
        #    loss = loss + regularizer_fn(model, rest_embed_support, stim_b)
        loss.backward()
        optimizer_inner.clip_grad(1.0)
        optimizer_inner.step()

    return {
        session_id: es.detach() for session_id, es in embeddings_stim.items()
    }  # stop gradient for the outer step


def inner_update_maml(
    model: nn.Module,
    loss_fn,
    stiminds_support: torch.Tensor,
    rest_embed_support: torch.Tensor,
    inputs_support: torch.Tensor,
    targets_support: torch.Tensor,
    stim_dim: int,
    inner_steps: int = 5,
    inner_lr: float = 5e-2,
    regularizer_fn=None,
) -> torch.Tensor:
    """
    MAML-style inner loop on stim_embed:
    - manually updates stim_embed with gradient steps using autograd.grad
    - create_graph=True enables outer backprop THROUGH these updates.

    Returns a differentiable stim_embed_final (no .detach()).
    """
    stim_embed = torch.zeros(
        1, stim_dim, device=stiminds_support.device, requires_grad=True
    )

    def inner_forward(stim_code: torch.Tensor) -> torch.Tensor:
        stim_b = stim_code.expand(stiminds_support.size(0), -1)
        preds = model(stiminds_support, rest_embed_support, stim_b)
        loss = loss_fn(preds, targets_support)
        if regularizer_fn is not None:
            loss = loss + regularizer_fn(model, rest_embed_support, stim_b)
        return loss

    stim_current = stim_embed
    for _ in range(inner_steps):
        loss_support = inner_forward(stim_current)
        grad = torch.autograd.grad(loss_support, stim_current, create_graph=True)[0]
        stim_current = (
            stim_current - inner_lr * grad
        )  # manual SGD step (differentiable)

    return stim_current  # keep graph for outer step (no detach)


def cache_rest_embeds(
    session_ids,
    data_dir="/home/mmattb/Projects/opto-coproc/data",
    out_dir="data",
    out_subdir="embedding_rest",
    quiet=False,
):
    for sidx, sid in enumerate(session_ids):
        if not quiet:
            print(f"{sid} ({sidx}/{len(session_ids)})")

        rest_dir = os.path.join(data_dir, sid, "torchbase")
        data_path = os.path.join(rest_dir, "base.torch")

        try:
            resting_data = torch.load(data_path)
        except OSError:
            # Not all sessions are usable; skip.
            continue

        # Ensure embedding cache dir
        _out_dir = os.path.join(out_dir, sid, out_subdir)
        os.makedirs(_out_dir, mode=0o777, exist_ok=True)

        embeddings_rest = embed_rest_sessions({sid: resting_data})[sid]
        torch.save(embeddings_rest, os.path.join(_out_dir, "er.torch"))


# TODO: trust region or other regularization for embed stim in addition to weight decay
