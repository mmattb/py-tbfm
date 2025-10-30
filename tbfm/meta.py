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
    cfg,
    inner_steps: int = None,
    quiet=True,
) -> torch.Tensor:
    """
    Optimize a per-episode stim_embed on SUPPORT ONLY.
    Does NOT backprop through the inner loop (two-backprop variant).
    Returns a detached stim_embed to use on the QUERY pass.

    Args:
        inner_steps: Optional override for number of inner steps.
                             If None, uses cfg.meta.training.inner_steps
    """
    # Extract parameters from config
    inner_steps = (
        inner_steps if inner_steps is not None else cfg.meta.training.inner_steps
    )
    lr = cfg.meta.training.optim.lr
    weight_decay = cfg.meta.training.optim.weight_decay
    grad_clip = cfg.training.grad_clip

    embed_dim_stim = model.model.bases.embed_dim_stim
    # Split out y values for loss fn

    with torch.no_grad():
        ys = {sid: d[2] for sid, d in data_support.items()}
        ys = model.norms(ys)
    device = ys[list(ys.keys())[0]].device

    embeddings_stim = {}
    for session_id in data_support.keys():
        emb = torch.randn(embed_dim_stim, device=device) * 0.1
        emb.requires_grad = True
        embeddings_stim[session_id] = emb

    optimizer_inner = utils.OptimCollection(
        {
            "meta": {
                "optimizers": [
                    torch.optim.AdamW((se,), lr=lr, weight_decay=weight_decay)
                    for se in embeddings_stim.values()
                ]
            }
        }
    )

    # Freeze model parameters during inner loop (stopgrad: no gradients w.r.t. model params)
    model.requires_grad_(False)
    model.eval()  # Disable dropout during inner loop optimization

    # Detach embeddings_rest to prevent graph building through it (we don't optimize it)
    embeddings_rest_detached = {
        sid: emb.detach() for sid, emb in embeddings_rest.items()
    }

    losses = []
    for eidx in range(inner_steps):
        optimizer_inner.zero_grad()

        # Clear any stored state from previous forward passes
        model.model.reset_state()

        preds = model(
            data_support,
            embeddings_rest=embeddings_rest_detached,
            embeddings_stim=embeddings_stim,
        )
        loss = 0
        for session_id, y in ys.items():
            loss += nn.MSELoss()(preds[session_id], y)
        loss /= len(preds)

        if (eidx % 100) == 0 and not quiet:
            print(eidx, loss.item())
            losses.append((eidx, loss.item()))

        # Optional L2 penalty to maintain a small sorta trust region
        # Using mean() for dimension normalization so penalty is scale-invariant
        lambda_l2 = cfg.meta.training.lambda_l2
        if lambda_l2:
            l2_reg = 0.0
            for emb in embeddings_stim.values():
                l2_reg = l2_reg + (emb**2).mean()
            l2_reg /= len(embeddings_stim)
            loss = loss + (lambda_l2 * l2_reg)

        # Only gradients w.r.t. embeddings_stim (model params frozen, embeddings_rest detached)
        loss.backward()

        optimizer_inner.clip_grad(grad_clip)
        optimizer_inner.step()

    # Re-enable gradients for model parameters (for outer loop)
    model.requires_grad_(True)

    embeddings_stim = {
        session_id: es.detach() for session_id, es in embeddings_stim.items()
    }
    if quiet:
        return embeddings_stim
    return embeddings_stim, losses


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
