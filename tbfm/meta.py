import os

import torch
from torch import nn
from typing import Dict, Tuple, List

from . import acf
from . import utils

from ._multisession_module import TBFMMultisession


# Basis Residual Network ------------------------------------------------------


class BasisResidualNet(nn.Module):
    """
    Small MLP that maps stim embeddings to low-rank basis residuals.

    Architecture:
        embed_stim -> MLP -> basis_residual_rank -> W -> trial_len * num_bases

    The output is reshaped to (batch, trial_len, num_bases) and added to base bases.
    """

    def __init__(
        self,
        embed_dim_stim: int,
        basis_residual_rank: int,
        trial_len: int,
        num_bases: int,
        residual_mlp_hidden: int = 16,
        device=None,
    ):
        super().__init__()

        self.embed_dim_stim = embed_dim_stim
        self.basis_residual_rank = basis_residual_rank
        self.trial_len = trial_len
        self.num_bases = num_bases

        # Small MLP: embed_stim -> hidden -> rank
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim_stim, residual_mlp_hidden),
            nn.Tanh(),
            nn.Linear(residual_mlp_hidden, basis_residual_rank),
        )

        # Tall skinny projection: rank -> trial_len * num_bases
        self.W = nn.Linear(basis_residual_rank, trial_len * num_bases, bias=False)

        # Initialize W with small weights to start with near-zero residuals
        nn.init.normal_(self.W.weight, mean=0, std=0.01)

        self.to(device)

    def forward(self, embedding_stim: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding_stim: (batch, embed_dim_stim) or (embed_dim_stim,)
                           If 1D, will be expanded to batch dimension

        Returns:
            residual: (batch, trial_len, num_bases)
        """
        # Handle both batched and single embedding
        if embedding_stim.dim() == 1:
            embedding_stim = embedding_stim.unsqueeze(0)

        # MLP: (batch, embed_dim_stim) -> (batch, rank)
        h = self.mlp(embedding_stim)

        # Project: (batch, rank) -> (batch, trial_len * num_bases)
        residual_flat = self.W(h)

        # Reshape: (batch, trial_len * num_bases) -> (batch, trial_len, num_bases)
        residual = residual_flat.unflatten(
            dim=1, sizes=(self.trial_len, self.num_bases)
        )

        return residual


# Hypernetwork Meta-Learning ------------------------------------------------------


class SupportSetEncoder(nn.Module):
    """
    Encodes a support set into a fixed-size context vector.

    Takes predictions and targets from support set and aggregates them
    into a context representation for the hypernetwork.
    """

    def __init__(
        self,
        input_dim: int,  # Dimension of each support example feature
        hidden_dim: int = 64,
        context_dim: int = 32,
        use_attention: bool = False,
        num_heads: int = 4,
        device=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.use_attention = use_attention

        # Encode individual support examples
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        if use_attention:
            # Multi-head attention for aggregating support examples
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            )
            # Learnable query for attention
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Project to context dimension
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_features: (batch_size, input_dim) - features from support set
                             Can be statistics, prediction errors, etc.

        Returns:
            context: (context_dim,) - aggregated context vector
        """
        # Encode each support example
        # support_features: (n_support, input_dim)
        encoded = self.encoder(support_features)  # (n_support, hidden_dim)

        if self.use_attention:
            # Use attention to aggregate
            batch_size = encoded.shape[0]
            query = self.query.expand(batch_size, -1, -1)  # (n_support, 1, hidden_dim)

            # Attention: query attends to encoded support examples
            aggregated, _ = self.attention(
                query, encoded.unsqueeze(1), encoded.unsqueeze(1)
            )  # (n_support, 1, hidden_dim)
            aggregated = aggregated.squeeze(1)  # (n_support, hidden_dim)
            aggregated = aggregated.mean(dim=0)  # (hidden_dim,)
        else:
            # Simple mean aggregation
            aggregated = encoded.mean(dim=0)  # (hidden_dim,)

        # Project to context dimension
        context = self.projector(aggregated)  # (context_dim,)

        return context


class HyperResidualNet(nn.Module):
    """
    Hypernetwork that generates parameters for BasisResidualNet from support context.

    Instead of optimizing embeddings via gradient descent (MAML), this network
    takes a support set context and directly generates the residual network parameters.

    Architecture:
        support_context -> HyperNet -> BasisResidualNet parameters
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim_stim: int,  # For compatibility, but we generate residual params directly
        basis_residual_rank: int,
        trial_len: int,
        num_bases: int,
        residual_mlp_hidden: int = 16,
        hypernet_hidden: int = 128,
        device=None,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim_stim = embed_dim_stim
        self.basis_residual_rank = basis_residual_rank
        self.trial_len = trial_len
        self.num_bases = num_bases
        self.residual_mlp_hidden = residual_mlp_hidden

        # Calculate parameter counts for residual network
        # Residual MLP has: embed_stim -> hidden -> rank
        self.mlp_params_count = (
            embed_dim_stim * residual_mlp_hidden + residual_mlp_hidden +  # Layer 1 + bias
            residual_mlp_hidden * basis_residual_rank + basis_residual_rank  # Layer 2 + bias
        )

        # W matrix: rank -> trial_len * num_bases (no bias)
        self.W_params_count = basis_residual_rank * trial_len * num_bases

        self.total_params = self.mlp_params_count + self.W_params_count

        # Hypernetwork: context -> residual network parameters
        self.hypernet = nn.Sequential(
            nn.Linear(context_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, self.total_params),
        )

        # Initialize to generate small residual parameters (near-zero initialization)
        nn.init.normal_(self.hypernet[-1].weight, mean=0, std=0.001)
        nn.init.zeros_(self.hypernet[-1].bias)

        self.to(device)

    def forward(
        self,
        context: torch.Tensor,
        embedding_stim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate residual basis corrections from support context.

        Args:
            context: (context_dim,) - encoded support set context
            embedding_stim: (embed_dim_stim,) or (batch, embed_dim_stim) -
                           task embedding (used as input to generated residual net)

        Returns:
            residual: (batch, trial_len, num_bases) - basis residuals
        """
        # Generate parameters
        params = self.hypernet(context)  # (total_params,)

        # Split into MLP and W parameters
        mlp_params = params[:self.mlp_params_count]
        W_params = params[self.mlp_params_count:]

        # Parse MLP parameters
        offset = 0
        # Layer 1: embed_stim -> hidden
        w1_size = self.embed_dim_stim * self.residual_mlp_hidden
        w1 = mlp_params[offset:offset + w1_size].view(
            self.residual_mlp_hidden, self.embed_dim_stim
        )
        offset += w1_size
        b1 = mlp_params[offset:offset + self.residual_mlp_hidden]
        offset += self.residual_mlp_hidden

        # Layer 2: hidden -> rank
        w2_size = self.residual_mlp_hidden * self.basis_residual_rank
        w2 = mlp_params[offset:offset + w2_size].view(
            self.basis_residual_rank, self.residual_mlp_hidden
        )
        offset += w2_size
        b2 = mlp_params[offset:offset + self.basis_residual_rank]

        # W matrix: rank -> trial_len * num_bases
        W = W_params.view(self.trial_len * self.num_bases, self.basis_residual_rank)

        # Apply generated residual network
        # Handle both batched and single embedding
        if embedding_stim.dim() == 1:
            embedding_stim = embedding_stim.unsqueeze(0)  # (1, embed_dim_stim)

        batch_size = embedding_stim.shape[0]

        # Forward through generated MLP
        h = torch.nn.functional.linear(embedding_stim, w1, b1)  # (batch, hidden)
        h = torch.tanh(h)
        h = torch.nn.functional.linear(h, w2, b2)  # (batch, rank)

        # Forward through generated W
        residual_flat = torch.nn.functional.linear(h, W)  # (batch, trial_len * num_bases)

        # Reshape to (batch, trial_len, num_bases)
        residual = residual_flat.unflatten(
            dim=1, sizes=(self.trial_len, self.num_bases)
        )

        return residual


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


def hypernetwork_adapt(
    model: nn.Module,
    data_support,
    embeddings_rest,
    cfg,
    quiet=True,
) -> Dict[str, torch.Tensor]:
    """
    Fast adaptation using hypernetwork (single forward pass).
    Generates support context and produces adapted parameters directly.

    Args:
        model: TBFMMultisession model with hypernetwork components
        data_support: Support data dict {session_id: (runway, stiminds, y)}
        embeddings_rest: Rest embeddings dict {session_id: embedding}
        cfg: Config object
        quiet: If True, suppress progress messages

    Returns:
        support_contexts: Dict mapping session_id -> context vector
    """
    device = model.device

    # Normalize targets for computing support statistics
    with torch.no_grad():
        ys = {sid: d[2] for sid, d in data_support.items()}
        ys_norm = model.norms(ys)

    model.eval()  # Ensure model is in eval mode for forward pass

    support_contexts = {}

    for session_id, data in data_support.items():
        runway, stiminds, y = data
        y_norm = ys_norm[session_id]

        # Get embedding_rest for this session
        embedding_rest = embeddings_rest[session_id]

        # Forward pass to get predictions with rest embedding only
        # We'll use a dummy embedding_stim initially
        dummy_embed_stim = torch.zeros(
            model.model.bases.embed_dim_stim, device=device
        )

        with torch.no_grad():
            # Create temporary data dict for this session
            session_data = {session_id: (runway, stiminds, y)}
            session_embeddings_rest = {session_id: embedding_rest}
            session_embeddings_stim = {session_id: dummy_embed_stim}

            # Get predictions
            y_pred = model(
                session_data,
                embeddings_rest=session_embeddings_rest,
                embeddings_stim=session_embeddings_stim,
            )

            y_pred_session = y_pred[session_id]

            # Compute prediction errors and aggregate statistics
            # Features: [prediction_error, y_norm_mean, y_norm_std, y_pred_mean, y_pred_std]
            errors = (y_pred_session - y_norm).flatten(end_dim=-2)  # (batch*time, channels)

            # Aggregate statistics across batch and time
            error_mean = errors.mean(dim=0)  # (channels,)
            error_std = errors.std(dim=0)  # (channels,)
            y_norm_flat = y_norm.flatten(end_dim=-2)
            y_mean = y_norm_flat.mean(dim=0)
            y_std = y_norm_flat.std(dim=0)
            y_pred_flat = y_pred_session.flatten(end_dim=-2)
            y_pred_mean = y_pred_flat.mean(dim=0)
            y_pred_std = y_pred_flat.std(dim=0)

            # Concatenate all statistics as support features
            # Shape: (n_features, channels) where n_features=5
            support_features = torch.stack([
                error_mean, error_std, y_mean, y_std, y_pred_mean, y_pred_std
            ], dim=0)  # (6, channels)

            # Flatten for encoder input
            support_features_flat = support_features.flatten()  # (6 * channels,)

            # Encode to context
            # Get the support encoder from the model
            support_encoder = model.model.bases.support_encoder
            context = support_encoder(support_features_flat.unsqueeze(0)).squeeze(0)

            support_contexts[session_id] = context

            if not quiet:
                print(f"Encoded support for {session_id}: context shape {context.shape}")

    return support_contexts


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
