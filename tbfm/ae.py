import inspect
import hydra.utils
import torch
import torch.nn as nn
from typing import Optional, Union

from .utils import SessionDispatcher, rotate_session_from_batch


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
        use_lora: enable rank-1 LoRA on the selected session encoder matrix (lightweight)
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        use_bias: bool = False,
        use_lora: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.use_lora = use_lora
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

        # Rank-1 LoRA (global directions); per-session you’ll scale with a scalar alpha if you want
        if use_lora:
            self.a = nn.Parameter(torch.zeros(latent_dim).to(device))  # left direction
            self.b = nn.Parameter(torch.zeros(in_dim).to(device))  # right direction
            # init tiny so base dominates
            nn.init.normal_(self.a, std=1e-3)
            nn.init.normal_(self.b, std=1e-3)

    def pca_warm_start(
        self,
        x: torch.Tensor,  # [B, T, C]
        mask: torch.Tensor,  # []     (long indices into [0..C_max-1])
        center: str = "median",  # "median" or "mean"
        whiten: bool = False,  # scale by 1/singular_value
        eps: float = 1e-8,
        keep_rest: bool = True,  # keep other columns of W_enc unchanged
    ):
        """
        Initialize the session slice of W_enc with PCA directions:
          W_enc[:, mask] <- V_k^T  (rows are principal axes)
        If whiten=True, rows are scaled by 1/sigma_k to give unit-variance components.

        Notes:
        - Other columns of W_enc are left as-is (keep_rest=True). Set keep_rest=False if
          you want to zero them first (rarely needed).
        """
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

        # SVD: Xc = U S Vh  (Vh: [min(N,C_s), C_s])
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        # Principal axes (columns of V): V = Vh^T, take top-k
        C_s = Xs.shape[1]
        k = min(self.latent_dim, C_s)

        V_top = Vh[:k, :]  # [k, C_s]  (rows are top-k PCs^T)
        if whiten:
            # Whitening: scale each row by 1/sigma
            S_top = S[:k].clamp_min(eps)  # [k]
            V_top = V_top / S_top.unsqueeze(1)

        # Optionally zero the slice first
        if not keep_rest:
            with torch.no_grad():
                self.w_enc[:, mask] = 0.0

        # Write PCA rows into the session slice
        with torch.no_grad():
            self.w_enc[:k, mask] = V_top  # rows 0..k-1 get PCA directions
            # (If self.C_star > k, we leave remaining rows as-is.)

        # Optional: row-orthonormalize the *written* rows of the slice for numerical niceness
        with torch.no_grad():
            W_slice = self.w_enc[:k, mask]  # [k, C_s]
            # Orthonormalize rows via QR on transpose (columns orthonormal -> rows orthonormal after T)
            Q, _ = torch.linalg.qr(W_slice.t(), mode="reduced")  # [C_s, k]
            self.w_enc[:k, mask] = Q.t()  # [k, C_s]

    def identity_warm_start(self):
        with torch.no_grad():
            d1, d2 = self.w_enc.shape
            m_dim = max(d1, d2)
            i = torch.eye(m_dim)
            self.w_enc[:] = i[:d1, :d2]

    @staticmethod
    def _indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
        # mask: (in_dim,) bool → indices where True
        if mask.dtype == torch.bool:
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)
        elif mask.dtype in (torch.int32, torch.int64):
            return mask
        else:
            raise ValueError("mask must be bool or int index tensor")

    def session_mats(
        self,
        mask: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ):
        """
        Build session-specific encoder/decoder matrices by selecting present channels.
        Optionally add rank-1 LoRA: W_s = w_sel + alpha * a_sel_outer

        Returns:
            w_enc_s: (latent_dim, C_s)
            w_dec_s: (C_s, latent_dim)  (tied transpose)
        """
        if isinstance(mask, list):
            idx = torch.tensor(mask, device=self.w_enc.device, dtype=torch.long)
        else:
            idx = self._indices_from_mask(mask).to(self.w_enc.device)

        w_sel = self.w_enc[:, idx]  # (latent_dim, C_s); selected dims only

        if self.use_lora and lora_alpha is not None:
            # LoRA rank-1 update on the selected columns
            a = self.a.view(self.latent_dim, 1)  # (latent_dim, 1)
            b_sel = self.b[idx].view(1, -1)  # (1, C_s)
            w_sel = w_sel + lora_alpha * (a @ b_sel)

        # Tied decoder
        w_enc_s = w_sel
        w_dec_s = w_sel.t().contiguous()  # (C_s, latent_dim)
        return w_enc_s, w_dec_s, idx

    def encode(
        self,
        x: torch.Tensor,
        mask: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (..., C_s)
        returns h: (..., latent_dim)
        """
        w_enc_s, _, _ = self.session_mats(mask, lora_alpha)
        h = x @ w_enc_s.t()  # (..., C_s) @ (C_s, latent_dim) -> (..., latent_dim)
        if self.b_enc is not None:
            h = h + self.b_enc
        return h

    def decode(
        self,
        h: torch.Tensor,
        mask: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h: (..., latent_dim)
        returns x_hat: (..., C_s)
        """
        _, w_dec_s, _ = self.session_mats(mask, lora_alpha)
        x_hat = h @ w_dec_s.t()  # (..., latent_dim) @ (latent_dim, C_s) -> (..., C_s)
        return x_hat

    def reconstruct(
        self,
        x: torch.Tensor,
        mask: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decode(self.encode(x, mask, lora_alpha), mask, lora_alpha)

    @staticmethod
    def orthogonality_penalty(w_enc_s: torch.Tensor) -> torch.Tensor:
        """
        Soft orthogonality on encoder rows (latent axes), promoting identifiability.
        w_enc_s: (latent_dim, C_s)
        """
        # Normalize rows
        R = w_enc_s / (w_enc_s.norm(dim=1, keepdim=True) + 1e-8)
        G = R @ R.t()  # (latent_dim, latent_dim)
        I = torch.eye(G.size(0), device=G.device).to(self.device)
        return ((G - I) ** 2).sum()

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
        "session_mats",
    ]

    def register_closure(self, name, values):
        for dm in SessionDispatcherLinearAE.DISPATCH_METHODS:
            sig = inspect.signature(getattr(LinearChannelAE, dm))
            if name in sig.parameters:
                self.close_kwarg(dm, name, values)

    def register_lora_alphas(self, lora_alphas: dict):
        return self.register_closure("lora_alpha", lora_alphas)

    def register_masks(self, masks: dict):
        return self.register_closure("mask", masks)


def dispatch_warm_start(aes, masks, data, is_identity=False, device=None):
    for session_id in data.keys():
        if not isinstance(data, dict):
            d = rotate_session_from_batch(data, session_id, device=device)
        else:
            d = data[session_id]

        if is_identity:
            aes.instances[session_id].identity_warm_start()
        else:
            mask = masks[session_id]
            x = torch.cat((d[0], d[2]), dim=1)  # 20, 164 -> 184
            aes.instances[session_id].pca_warm_start(x, mask=mask)


def make_masks(data, device=None):
    masks = {}

    for session_id in data.session_ids:
        masks[session_id] = torch.arange(data.get_session_num_feats(session_id)).to(
            device
        )
    return masks


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
    datas: {session_id: data (batch, time, ch)}
    """
    in_dims = {
        session_id: data.get_session_num_feats(session_id)
        for session_id in data.session_ids
    }

    aes = from_cfg_and_in_dims(cfg, in_dims, shared=shared, device=device, **kwargs)
    masks = make_masks(data, device=device)
    aes.register_masks(masks)

    if warm_start:
        dispatch_warm_start(
            aes, masks, data, device=device, is_identity=cfg.ae.warm_start_is_identity
        )

    return aes
