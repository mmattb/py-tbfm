import torch
import torch.nn as nn
from typing import Optional, Union


class LinearChannelAE(nn.Module):
    """
    Linear channel autoencoder with tied weights for variable-channel sessions.
    - Global encoder weight: w_enc in R[latent_dim, num_chan_max]
    - Decoder tied as w_dec = w_enc.T
    - For a session with present channels given by mask or indices, we select columns/rows.

    Args:
        num_chan_max: maximum number of channels across sessions
        latent_dim: latent width (e.g., 16–64)
        use_bias: if True, learn encoder bias (decoder bias tied to zero by default)
        use_lora: enable rank-1 LoRA on the selected session encoder matrix (lightweight)
    """

    def __init__(
        self,
        num_chan_max: int,
        latent_dim: int,
        use_bias: bool = True,
        use_lora: bool = False,
        device=None,
    ):
        super().__init__()
        self.num_chan_max = num_chan_max
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.use_lora = use_lora
        self.device = device

        # Encoder weight (global). Kaiming uniform works fine here.
        self.w_enc = nn.Parameter(torch.empty(latent_dim, num_chan_max).to(device))
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
            self.b = nn.Parameter(
                torch.zeros(num_chan_max).to(device)
            )  # right direction
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

    @staticmethod
    def _indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
        # mask: (num_chan_max,) bool → indices where True
        if mask.dtype == torch.bool:
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)
        elif mask.dtype in (torch.int32, torch.int64):
            return mask
        else:
            raise ValueError("mask must be bool or int index tensor")

    def session_mats(
        self,
        mask_or_idx: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ):
        """
        Build session-specific encoder/decoder matrices by selecting present channels.
        Optionally add rank-1 LoRA: W_s = w_sel + alpha * a_sel_outer

        Returns:
            w_enc_s: (latent_dim, C_s)
            w_dec_s: (C_s, latent_dim)  (tied transpose)
        """
        if isinstance(mask_or_idx, list):
            idx = torch.tensor(mask_or_idx, device=self.w_enc.device, dtype=torch.long)
        else:
            idx = self._indices_from_mask(mask_or_idx).to(self.w_enc.device)

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
        mask_or_idx: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (..., C_s)
        returns h: (..., latent_dim)
        """
        w_enc_s, _, _ = self.session_mats(mask_or_idx, lora_alpha)
        h = x @ w_enc_s.t()  # (..., C_s) @ (C_s, latent_dim) -> (..., latent_dim)
        if self.b_enc is not None:
            h = h + self.b_enc
        return h

    def decode(
        self,
        h: torch.Tensor,
        mask_or_idx: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h: (..., latent_dim)
        returns x_hat: (..., C_s)
        """
        _, w_dec_s, _ = self.session_mats(mask_or_idx, lora_alpha)
        x_hat = h @ w_dec_s.t()  # (..., latent_dim) @ (latent_dim, C_s) -> (..., C_s)
        return x_hat

    def reconstruct(
        self,
        x: torch.Tensor,
        mask_or_idx: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decode(
            self.encode(x, mask_or_idx, lora_alpha), mask_or_idx, lora_alpha
        )

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


class LinearAEMaskedDispatcher(nn.Module):
    """
    Thin wrapper around LinearAE which binds the mask arguments statically.
    """

    def __init__(self, lae, mask, lora_alpha=None):
        super().__init__()
        self.lae = lae
        self.mask = mask
        self.lora_alpha = lora_alpha

    def encode(self, x):
        return self.lae.encode(x, self.mask, lora_alpha=self.lora_alpha)

    def decode(self, z):
        return self.lae.decode(z, self.mask, lora_alpha=self.lora_alpha)

    def forward(self, *args, **kwargs):
        return self.lae(*args, **kwargs)
