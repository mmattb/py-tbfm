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
                self.w_enc[:, mask] = 0.0

            # Write PCA rows into the session slice
            self.w_enc[:k, mask] = V_top  # rows 0..k-1 get PCA directions
            # (If self.C_star > k, we leave remaining rows as-is.)

            # Optional: row-orthonormalize the *written* rows of the slice for numerical niceness
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
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
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


class TwoStageAffineAE(nn.Module):
    """
    Two-stage linear autoencoder with session-specific adapters and shared encoder/decoder.

    Architecture:
        x_s ∈ ℝ^{d_s} → A_s (adapter) → h_s ∈ ℝ^{d_c} → W_e (encoder) → z ∈ ℝ^{d_z}
        z → W_d (decoder) → h_s → A_s^T → x̂_s

    Args:
        session_dims: dict mapping session_id → input dimensionality
        canonical_dim: intermediate canonical dimension (d_c)
        latent_dim: latent dimension (d_z)
        use_adapter_bias: whether adapters have bias terms
        use_encoder_bias: whether shared encoder/decoder have bias terms
    """

    def __init__(
        self,
        session_dims: dict,
        canonical_dim: int,
        latent_dim: int,
        use_adapter_bias: bool = True,
        use_encoder_bias: bool = True,
        device=None,
    ):
        super().__init__()
        self.session_dims = session_dims
        self.session_ids = list(session_dims.keys())
        self.canonical_dim = canonical_dim
        self.latent_dim = latent_dim
        self.use_adapter_bias = use_adapter_bias
        self.use_encoder_bias = use_encoder_bias
        self.device = device

        # Session-specific adapters: d_s → d_c
        self.adapters = nn.ModuleDict()
        for session_id, d_s in session_dims.items():
            adapter = nn.Linear(
                d_s, canonical_dim, bias=use_adapter_bias, device=device
            )
            # Orthogonal initialization for adapters
            nn.init.orthogonal_(adapter.weight)
            if use_adapter_bias:
                nn.init.zeros_(adapter.bias)
            self.adapters[session_id] = adapter

        # Shared encoder: d_c → d_z
        self.encoder = nn.Linear(
            canonical_dim, latent_dim, bias=use_encoder_bias, device=device
        )
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        if use_encoder_bias:
            nn.init.zeros_(self.encoder.bias)

    def encode(self, x, session_id=None):
        """
        Encode input through adapter then shared encoder.

        Args:
            x: input tensor (..., d_s) or dict {session_id: (..., d_s)}
            session_id: if x is tensor, which session it belongs to

        Returns:
            z: latent tensor (..., d_z) or dict {session_id: (..., d_z)}
        """
        if isinstance(x, dict):
            return {sid: self.encode(x_s, session_id=sid) for sid, x_s in x.items()}

        # x: (..., d_s) → adapter → (..., d_c)
        h = self.adapters[session_id](x)
        # h: (..., d_c) → encoder → (..., d_z)
        z = self.encoder(h)
        return z

    def decode(self, z, session_id=None):
        """
        Decode latent through shared decoder then adapter transpose.

        Decoder uses tied weights (W_d = W_e^T) and symmetric bias handling:
        - If encoder has bias b_e, decode as: h = W_e^T @ (z - b_e)
        - If adapter has bias b_a, we need to invert it: x = A^T @ (h - b_a)

        Args:
            z: latent tensor (..., d_z) or dict {session_id: (..., d_z)}
            session_id: if z is tensor, which session to decode to

        Returns:
            x_hat: reconstruction (..., d_s) or dict {session_id: (..., d_s)}
        """
        if isinstance(z, dict):
            return {sid: self.decode(z_s, session_id=sid) for sid, z_s in z.items()}

        # z: (..., d_z) → decoder (tied) → (..., d_c)
        # Symmetric bias handling: h = W_e^T @ (z - b_e)
        if self.encoder.bias is not None:
            z_unbiased = z - self.encoder.bias
            h = z_unbiased @ self.encoder.weight  # (z - b_e) @ W_e^T
        else:
            h = z @ self.encoder.weight  # z @ W_e^T

        # h: (..., d_c) → adapter^T → (..., d_s)
        # Invert adapter: x = A^T @ (h - b_a)
        adapter = self.adapters[session_id]
        if self.use_adapter_bias and adapter.bias is not None:
            h_unbiased = h - adapter.bias  # Remove bias before transpose
            x_hat = h_unbiased @ adapter.weight
        else:
            x_hat = h @ adapter.weight

        return x_hat

    def reconstruct(self, x, session_id=None):
        """Full reconstruction: encode then decode."""
        if isinstance(x, dict):
            return {
                sid: self.reconstruct(x_s, session_id=sid) for sid, x_s in x.items()
            }
        return self.decode(self.encode(x, session_id=session_id), session_id=session_id)

    def pca_warm_start(
        self,
        data,
        center: str = "median",
        whiten: bool = False,
        eps: float = 1e-8,
    ):
        """
        Initialize adapters and encoder with PCA.

        Strategy:
        1. For each session, run PCA on session data to initialize adapter
        2. Project all data through adapters to canonical space
        3. Run PCA on canonical representations to initialize shared encoder

        Args:
            data: dict {session_id: tensor [B, T, C_s]} or similar structure
            center: 'median' or 'mean' for centering
            whiten: whether to scale by inverse singular values
        """
        # Step 1: Initialize adapters with per-session PCA
        canonical_data = {}

        with torch.no_grad():
            for session_id in self.session_ids:
                # Extract session data
                if isinstance(data, dict) and session_id in data:
                    x_s = data[session_id]
                    if isinstance(x_s, (tuple, list)):
                        # Handle (runway, covariates, y) format
                        x_s = torch.cat(
                            [x_s[0], x_s[2]], dim=1
                        )  # concat runway + targets
                else:
                    # Might be a data object with rotation method
                    d = rotate_session_from_batch(data, session_id, device=self.device)
                    x_s = torch.cat((d[0], d[2]), dim=1)

                # Flatten to [N, C_s]
                x_s = x_s.flatten(end_dim=-2).to(
                    device=self.device, dtype=self.encoder.weight.dtype
                )

                # Center
                if center == "median":
                    c = x_s.median(dim=0).values
                elif center == "mean":
                    c = x_s.mean(dim=0)
                else:
                    raise ValueError("center must be 'median' or 'mean'")
                x_c = x_s - c

                # SVD for PCA
                U, S, Vh = torch.linalg.svd(x_c, full_matrices=False)
                k = min(self.canonical_dim, x_c.shape[1])

                # Principal components (rows of Vh are PCs in input space)
                V_top = Vh[:k, :]  # [k, C_s]
                if whiten:
                    S_top = S[:k].clamp_min(eps)
                    V_top = V_top / S_top.unsqueeze(1)

                # Initialize adapter weight with PCs
                self.adapters[session_id].weight[:k, :] = V_top
                # Orthonormalize in case k < canonical_dim
                if k < self.canonical_dim:
                    # Fill remaining rows with random orthogonal vectors
                    remaining = self.canonical_dim - k
                    rand_vecs = torch.randn(remaining, x_c.shape[1], device=self.device)
                    Q, _ = torch.linalg.qr(rand_vecs.T)
                    self.adapters[session_id].weight[k:, :] = Q.T[:remaining, :]

                # Project to canonical space for step 2
                h_s = self.adapters[session_id](x_s)
                canonical_data[session_id] = h_s

            # Step 2: Initialize shared encoder with PCA on canonical data
            h_all = torch.cat([canonical_data[sid] for sid in self.session_ids], dim=0)

            # Center
            if center == "median":
                c_h = h_all.median(dim=0).values
            elif center == "mean":
                c_h = h_all.mean(dim=0)
            h_centered = h_all - c_h

            # SVD
            U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
            k = min(self.latent_dim, self.canonical_dim)

            V_top = Vh[:k, :]  # [k, d_c]
            if whiten:
                S_top = S[:k].clamp_min(eps)
                V_top = V_top / S_top.unsqueeze(1)

            # Initialize encoder weight with PCA directions
            self.encoder.weight[:k, :] = V_top

            # Orthonormalize the rows we just initialized
            W_slice = self.encoder.weight[:k, :]  # [k, d_c]
            Q, _ = torch.linalg.qr(W_slice.T, mode="reduced")  # [d_c, k]
            self.encoder.weight[:k, :] = Q.T  # [k, d_c]
            # Remaining rows (if k < latent_dim) keep their Kaiming initialization

    def identity_warm_start(self):
        """Initialize with identity-like mappings (for debugging)."""
        with torch.no_grad():
            # Adapters: make them as close to identity as possible
            for session_id, d_s in self.session_dims.items():
                w = self.adapters[session_id].weight
                d_min = min(self.canonical_dim, d_s)
                w.zero_()
                w[:d_min, :d_min] = torch.eye(d_min, device=self.device)

            # Encoder: identity mapping
            d_min = min(self.latent_dim, self.canonical_dim)
            self.encoder.weight.zero_()
            self.encoder.weight[:d_min, :d_min] = torch.eye(d_min, device=self.device)

    @staticmethod
    def moment_matching_loss(z: torch.Tensor) -> tuple:
        """
        Compute moment matching losses to encourage standard Gaussian latent.

        L_mu = ||mean(z)||^2 / d_z  (mean squared deviation from zero, normalized)
        L_cov = ||cov(z) - I||^2 / d_z^2  (mean squared deviation from identity, normalized)

        Args:
            z: latent tensor [N, d_z]

        Returns:
            (L_mu, L_cov)
        """
        # Flatten batch/time dimensions if needed
        if z.ndim > 2:
            z = z.flatten(end_dim=-2)

        d_z = z.shape[1]

        # Zero mean loss (normalized by number of dimensions)
        mu = z.mean(dim=0)
        L_mu = (mu**2).sum() / d_z

        # Identity covariance loss (normalized by number of matrix elements)
        z_centered = z - mu
        N = z.shape[0]
        cov = (z_centered.T @ z_centered) / (N - 1)
        I = torch.eye(d_z, device=z.device, dtype=z.dtype)
        L_cov = ((cov - I) ** 2).sum() / (d_z * d_z)

        return L_mu, L_cov

    def adapter_orthogonality_penalty(self, session_id: str = None) -> torch.Tensor:
        """
        Compute ||A^T A - I||^2 to encourage orthogonal adapter columns.

        Args:
            session_id: specific session, or None for sum over all sessions
        """
        if session_id is not None:
            A = self.adapters[session_id].weight  # [d_c, d_s]
            G = A.T @ A  # [d_s, d_s]
            I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
            return ((G - I) ** 2).sum()
        else:
            # Sum over all sessions
            total = 0.0
            for sid in self.session_ids:
                total = total + self.adapter_orthogonality_penalty(sid)
            return total

    def get_optim(
        self,
        adapter_lr=1e-4,
        encoder_lr=1e-5,
        betas=(0.5, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
    ):
        """
        Create optimizer with separate learning rates for adapters vs shared encoder.
        """
        param_groups = [
            {
                "params": self.adapters.parameters(),
                "lr": adapter_lr,
            },
            {
                "params": [self.encoder.weight],
                "lr": encoder_lr,
            },
        ]

        if self.encoder.bias is not None:
            param_groups[1]["params"] = [self.encoder.weight, self.encoder.bias]

        return torch.optim.AdamW(
            param_groups,
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
        MSE reconstruction loss (same as LinearChannelAE for compatibility).
        """
        if mask_present is not None:
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
    Create autoencoder from config and data.
    Supports both single-stage (LinearChannelAE) and two-stage (TwoStageAffineAE).

    Args:
        cfg: hydra config with ae settings
        data: data object with session_ids and get_session_num_feats method
        shared: for single-stage, whether to share AE across sessions
        warm_start: whether to initialize with PCA
        device: torch device
    """
    # Check if using two-stage architecture
    use_two_stage = cfg.ae.get("use_two_stage", False)

    if use_two_stage:
        # Build two-stage AE
        session_dims = {
            session_id: data.get_session_num_feats(session_id)
            for session_id in data.session_ids
        }

        ae = TwoStageAffineAE(
            session_dims=session_dims,
            canonical_dim=cfg.ae.two_stage.canonical_dim,
            latent_dim=cfg.ae.module.latent_dim,
            use_adapter_bias=cfg.ae.two_stage.get("use_adapter_bias", True),
            use_encoder_bias=cfg.ae.two_stage.get("use_encoder_bias", True),
            device=device,
        )

        if warm_start:
            if cfg.ae.warm_start_is_identity:
                ae.identity_warm_start()
            else:
                # PCA warm-start on adapters and encoder
                ae.pca_warm_start(data)

        return ae

    else:
        # Original single-stage logic
        in_dims = {
            session_id: data.get_session_num_feats(session_id)
            for session_id in data.session_ids
        }

        aes = from_cfg_and_in_dims(cfg, in_dims, shared=shared, device=device, **kwargs)
        masks = make_masks(data, device=device)
        aes.register_masks(masks)

        if warm_start:
            dispatch_warm_start(
                aes,
                masks,
                data,
                device=device,
                is_identity=cfg.ae.warm_start_is_identity,
            )

        return aes
