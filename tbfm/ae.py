import hydra.utils
import torch
import torch.nn as nn
from typing import Optional, Union

from .utils import SessionDispatcher, iter_loader


class LinearChannelAE(nn.Module):
    """
    Linear channel autoencoder with tied weights for variable-channel sessions.
    - Global encoder weight: w_enc in R[latent_dim, in_dim]
    - Decoder tied as w_dec = w_enc.T
    - For a session with present channels given by mask or indices, we select columns/rows.

    Args:
        in_dim: maximum number of channels across sessions
        latent_dim: latent width (e.g., 16–64)
        use_bias: if True, learn encoder bias (decoder bias tied to zero by default)
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        use_bias: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.device = device

        # Encoder weight (global). Kaiming uniform works fine here.
        self.w_enc = nn.Parameter(torch.empty(latent_dim, in_dim).to(device))
        nn.init.kaiming_uniform_(self.w_enc, a=5**0.5)

        # Optional encoder bias (decoder remains tied = no extra bias by default)
        if use_bias:
            self.b_enc = (
                nn.Parameter(torch.zeros(latent_dim).to(device)) if use_bias else None
            )
        else:
            self.b_enc = None

    def pca_warm_start(
        self,
        x: torch.Tensor,  # [B, T, C]
        center: str = "median",  # "median" or "mean"
        whiten: bool = False,  # scale by 1/singular_value
        eps: float = 1e-8,
    ):
        """
        Initialize W_enc with PCA directions.
        If whiten=True, rows are scaled by 1/sigma_k to give unit-variance components.
        """
        with torch.no_grad():
            x = x.flatten(end_dim=1)
            device = self.w_enc.device
            dtype = self.w_enc.dtype

            Xs = x.to(device=device, dtype=dtype)  # [N, C]
            # Robust or standard centering
            if center == "median":
                c = Xs.median(dim=0).values
            elif center == "mean":
                c = Xs.mean(dim=0)
            else:
                raise ValueError("center must be 'median' or 'mean'")
            Xc = Xs - c

            # SVD: Xc = U S Vh  (Vh: [min(N,C), C])
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            # Principal axes (columns of V): V = Vh^T, take top-k
            C = Xs.shape[1]
            k = min(self.latent_dim, C)

            V_top = Vh[:k, :]  # [k, C]  (rows are top-k PCs^T)
            if whiten:
                # Whitening: scale each row by 1/sigma
                S_top = S[:k].clamp_min(eps)  # [k]
                V_top = V_top / S_top.unsqueeze(1)

            # Write PCA rows into W_enc
            self.w_enc[:k, :] = V_top  # rows 0..k-1 get PCA directions

            # Optional: row-orthonormalize for numerical stability
            W_slice = self.w_enc[:k, :]  # [k, C]
            Q, _ = torch.linalg.qr(W_slice.t(), mode="reduced")  # [C, k]
            self.w_enc[:k, :] = Q.t()  # [k, C]

    def identity_warm_start(self):
        with torch.no_grad():
            d1, d2 = self.w_enc.shape
            m_dim = max(d1, d2)
            i = torch.eye(m_dim)
            self.w_enc[:] = i[:d1, :d2]

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (..., C)
        returns h: (..., latent_dim)
        """
        w_enc = self.w_enc

        h = x @ w_enc.t()  # (..., C) @ (C, latent_dim) -> (..., latent_dim)
        if self.b_enc is not None:
            h = h + self.b_enc
        return h

    def decode(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (..., latent_dim)
        returns x_hat: (..., C)
        """
        w_enc = self.w_enc

        w_dec = w_enc.t().contiguous()  # Tied decoder
        x_hat = h @ w_dec.t()  # (..., latent_dim) @ (latent_dim, C) -> (..., C)
        return x_hat

    def reconstruct(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(x))

    def get_optim(
        self, lr=1e-4, betas=(0.5, 0.99), eps=1e-8, weight_decay=0.0, amsgrad=True
    ):
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mask_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE over present channels. If mask_present is None, averages over all dims.
        x, x_hat: (B, C_s) or (..., C_s)
        mask_present: optional bool mask of shape (C_s,) to exclude any channels from loss.
        """
        if mask_present is not None:
            # Broadcast mask over batch/time
            while mask_present.ndim < x.ndim:
                mask_present = mask_present.unsqueeze(0)
            diff2 = ((x - x_hat) ** 2) * mask_present
            denom = mask_present.sum().clamp_min(1.0)
            return diff2.sum() / denom
        else:
            return nn.functional.mse_loss(x_hat, x)


class SessionDispatcherLinearAE(SessionDispatcher):
    DISPATCH_METHODS = [
        "encode",
        "decode",
        "pca_warm_start",
        "reconstruct",
    ]


def dispatch_warm_start(aes, data, is_identity=False, device=None):
    if not isinstance(data, dict):
        # Materialise a single batch from a SessionLoader so we can index by session
        _, data = iter_loader(iter(data), data, device=device)
    for session_id in data.keys():
        d = data[session_id]

        if is_identity:
            aes.instances[session_id].identity_warm_start()
        else:
            x = torch.cat((d[0], d[2]), dim=1)  # runway + targets: 20, 164 -> 184
            aes.instances[session_id].pca_warm_start(x)


def from_cfg_single(cfg, in_dim, device=None, **kwargs):
    ae = hydra.utils.instantiate(cfg.ae.module, in_dim=in_dim, device=device, **kwargs)
    return ae


def from_cfg_and_in_dims(cfg, in_dims: dict, shared=False, device=None, **kwargs):
    """
    in_dims: {session_id: in_dim}
    """
    instances = {}  # {session_id, instance}

    if shared:
        in_dim = max(in_dims.values())
        instance = from_cfg_single(cfg, in_dim, device=device, **kwargs)
        for session_id, _in_dim in in_dims.items():
            instances[session_id] = instance
    else:
        for session_id, in_dim in in_dims.items():
            instance = from_cfg_single(cfg, in_dim, device=device, **kwargs)
            instances[session_id] = instance
    return hydra.utils.instantiate(cfg.ae.dispatcher, instances)


def from_cfg_and_data(cfg, data, shared=False, warm_start=True, device=None, **kwargs):
    """
    Create autoencoder from config and data.

    Args:
        cfg: hydra config with ae settings
        data: data object with session_ids and get_session_num_feats method
        shared: whether to share AE across sessions
        warm_start: whether to initialize with PCA
        device: torch device
    """
    in_dims = {
        session_id: data.get_session_num_feats(session_id)
        for session_id in data.session_ids
    }

    aes = from_cfg_and_in_dims(cfg, in_dims, shared=shared, device=device, **kwargs)

    if warm_start:
        dispatch_warm_start(
            aes,
            data,
            device=device,
            is_identity=cfg.ae.warm_start_is_identity,
        )

    return aes
