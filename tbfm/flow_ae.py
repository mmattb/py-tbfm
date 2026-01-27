"""
Normalizing Flow-based Autoencoder with support for variable-channel sessions.

Uses affine coupling layers (RealNVP-style) for exact invertibility.
"""

import inspect
import torch
import torch.nn as nn
from typing import Optional, Union, List

from .utils import SessionDispatcher, rotate_session_from_batch


class AffineCouplingLayer(nn.Module):
    """
    RealNVP-style affine coupling layer - invertible by construction.

    Split input x = [x1, x2], then:
    - Forward:  z1 = x1, z2 = x2 * exp(s(x1)) + t(x1)
    - Reverse:  x1 = z1, x2 = (z2 - t(z1)) * exp(-s(z1))

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer width for scale/translate networks (default: 64, reduced from 128)
        split_dim: Where to split the input (default: dim // 2)
        dropout: Dropout rate for regularization (default: 0.1)
        use_spectral_norm: Apply spectral normalization to weights (default: False)
        device: Device to place parameters on
    """

    def __init__(self, dim: int, hidden_dim: int = 64, split_dim: Optional[int] = None,
                 dropout: float = 0.1, use_spectral_norm: bool = False, device=None):
        super().__init__()
        self.dim = dim
        self.split_dim = split_dim or (dim // 2)
        self.dim2 = dim - self.split_dim
        self.device = device
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm

        # Helper to optionally apply spectral normalization
        def maybe_spectral_norm(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Scale network (bounded output for stability)
        self.scale_net = nn.Sequential(
            maybe_spectral_norm(nn.Linear(self.split_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            maybe_spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            maybe_spectral_norm(nn.Linear(hidden_dim, self.dim2)),
            nn.Tanh()  # Keep in [-1, 1]
        )

        # Translation network
        self.translate_net = nn.Sequential(
            maybe_spectral_norm(nn.Linear(self.split_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            maybe_spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            maybe_spectral_norm(nn.Linear(hidden_dim, self.dim2))
        )

        # Initialize to near-identity
        for layer in [self.scale_net[-2], self.translate_net[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Move to device
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, reverse: bool = False, return_log_det: bool = False):
        """
        Args:
            x: Input tensor (..., dim)
            reverse: If True, compute inverse transformation
            return_log_det: If True, return (output, log_det_jacobian)

        Returns:
            output: Transformed tensor (..., dim)
            log_det: Log determinant (only if return_log_det=True and reverse=False)
        """
        x1, x2 = x.split([self.split_dim, self.dim2], dim=-1)

        if not reverse:
            # Forward pass: encode
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            z2 = x2 * torch.exp(s) + t
            z = torch.cat([x1, z2], dim=-1)

            if return_log_det:
                log_det = s.sum(dim=-1)
                return z, log_det
            return z
        else:
            # Reverse pass: decode (exact inverse)
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            x2_recon = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, x2_recon], dim=-1)


class FlowChannelAE(nn.Module):
    """
    Normalizing flow autoencoder with support for variable-channel sessions.

    Architecture:
    1. Linear projection: in_dim -> latent_dim (tied weights for decode)
    2. Normalizing flow in latent space (fully invertible)

    Args:
        in_dim: Maximum number of input channels
        latent_dim: Latent space dimension
        num_flow_layers: Number of coupling layers (default: 3, reduced from 4)
        hidden_dim: Hidden layer width in coupling layers (default: 64, reduced from 128)
        dropout: Dropout rate in coupling layers (default: 0.1)
        use_spectral_norm: Apply spectral normalization to coupling layer weights (default: False)
        latent_noise_scale: Scale of Gaussian noise added to latents during training (default: 0.0)
        use_linear_projection: If True, use linear projection to latent_dim first
        use_lora: If True, enable LoRA adaptation (for API compatibility; not yet implemented)
        device: Device to place parameters on
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        num_flow_layers: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_spectral_norm: bool = False,
        latent_noise_scale: float = 0.0,
        use_linear_projection: bool = True,
        use_lora: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_flow_layers = num_flow_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm
        self.latent_noise_scale = latent_noise_scale
        self.use_linear_projection = use_linear_projection
        self.use_lora = use_lora
        self.device = device

        # Optional linear projection layer (for dimensionality reduction)
        if use_linear_projection and latent_dim != in_dim:
            self.w_proj = nn.Parameter(torch.empty(latent_dim, in_dim).to(device))
            nn.init.orthogonal_(self.w_proj)
        else:
            self.w_proj = None
            # If no projection, latent_dim must equal in_dim for flows
            if latent_dim != in_dim:
                raise ValueError(
                    "If use_linear_projection=False, latent_dim must equal in_dim"
                )

        # Normalizing flow layers
        flow_dim = latent_dim
        self.flows = nn.ModuleList([
            AffineCouplingLayer(flow_dim, hidden_dim, dropout=dropout,
                              use_spectral_norm=use_spectral_norm, device=device)
            for _ in range(num_flow_layers)
        ])

        # Permutation indices for mixing between layers
        perms = torch.stack([torch.randperm(flow_dim) for _ in range(num_flow_layers)])
        if device is not None:
            perms = perms.to(device)
        self.register_buffer('permutations', perms)

    def pca_warm_start(
        self,
        x: torch.Tensor,  # [B, T, C]
        mask: torch.Tensor,
        center: str = "median",
        whiten: bool = False,
        eps: float = 1e-8,
        keep_rest: bool = True,
    ):
        """
        Initialize projection matrix with PCA directions.
        Only applicable if use_linear_projection=True.
        """
        if not self.use_linear_projection or self.w_proj is None:
            return

        x = x.flatten(end_dim=1)
        device = self.w_proj.device
        dtype = self.w_proj.dtype

        Xs = x.to(device=device, dtype=dtype)  # [N, C]

        # Center
        if center == "median":
            c = Xs.median(dim=0).values
        elif center == "mean":
            c = Xs.mean(dim=0)
        else:
            raise ValueError("center must be 'median' or 'mean'")
        Xc = Xs - c

        # SVD
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        C_s = Xs.shape[1]
        k = min(self.latent_dim, C_s)

        V_top = Vh[:k, :]  # [k, C_s]
        if whiten:
            S_top = S[:k].clamp_min(eps)
            V_top = V_top / S_top.unsqueeze(1)

        if not keep_rest:
            with torch.no_grad():
                self.w_proj[:, mask] = 0.0

        with torch.no_grad():
            self.w_proj[:k, mask] = V_top

            # Orthonormalize
            W_slice = self.w_proj[:k, mask]
            Q, _ = torch.linalg.qr(W_slice.t(), mode="reduced")
            self.w_proj[:k, mask] = Q.t()

    def identity_warm_start(self):
        """Initialize projection to identity (or closest possible)."""
        if self.w_proj is not None:
            with torch.no_grad():
                d1, d2 = self.w_proj.shape
                m_dim = max(d1, d2)
                i = torch.eye(m_dim, device=self.w_proj.device)
                self.w_proj[:] = i[:d1, :d2]

    @staticmethod
    def _indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)
        elif mask.dtype in (torch.int32, torch.int64):
            return mask
        else:
            raise ValueError("mask must be bool or int index tensor")

    def _project_down(
        self,
        x: torch.Tensor,
        mask: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """Project from input space to latent space using session mask."""
        if self.w_proj is None:
            return x

        if isinstance(mask, list):
            idx = torch.tensor(mask, device=self.w_proj.device, dtype=torch.long)
        else:
            idx = self._indices_from_mask(mask).to(self.w_proj.device)

        w_sel = self.w_proj[:, idx]  # [latent_dim, C_s]
        return x @ w_sel.t()  # (..., C_s) @ (C_s, latent_dim) -> (..., latent_dim)

    def _project_up(
        self,
        h: torch.Tensor,
        mask: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """Project from latent space back to input space using session mask."""
        if self.w_proj is None:
            return h

        if isinstance(mask, list):
            idx = torch.tensor(mask, device=self.w_proj.device, dtype=torch.long)
        else:
            idx = self._indices_from_mask(mask).to(self.w_proj.device)

        w_sel = self.w_proj[:, idx]  # [latent_dim, C_s]
        return h @ w_sel  # (..., latent_dim) @ (latent_dim, C_s) -> (..., C_s)

    def _apply_flow(
        self,
        h: torch.Tensor,
        reverse: bool = False,
        return_log_det: bool = False,
    ):
        """Apply normalizing flow transformations."""
        if not reverse:
            # Encode: apply flows forward
            log_det_total = 0
            for i, flow in enumerate(self.flows):
                perm = self.permutations[i]
                h = h[..., perm]  # Permute
                if return_log_det:
                    h, log_det = flow(h, reverse=False, return_log_det=True)
                    log_det_total = log_det_total + log_det
                else:
                    h = flow(h, reverse=False, return_log_det=False)

            if return_log_det:
                return h, log_det_total
            return h
        else:
            # Decode: apply flows backward
            for i in reversed(range(len(self.flows))):
                flow = self.flows[i]
                perm = self.permutations[i]
                h = flow(h, reverse=True)
                # Inverse permutation
                inv_perm = torch.argsort(perm)
                h = h[..., inv_perm]
            return h

    def encode(
        self,
        x: torch.Tensor,
        mask: Union[torch.Tensor, list],
        return_log_det: bool = False,
        add_noise: bool = None,
    ) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor (..., C_s) where C_s is session channel count
            mask: Channel mask or indices for this session
            return_log_det: If True, return (z, log_det_jacobian)
            add_noise: If True, add Gaussian noise to latents (uses self.training if None)

        Returns:
            z: Latent representation (..., latent_dim)
            log_det: Log determinant (only if return_log_det=True)
        """
        # Project to latent dimension
        h = self._project_down(x, mask)

        # Apply normalizing flows
        if return_log_det:
            z, log_det = self._apply_flow(h, reverse=False, return_log_det=True)
            return z, log_det
        else:
            z = self._apply_flow(h, reverse=False, return_log_det=False)

            # Add noise during training for regularization
            if add_noise is None:
                add_noise = self.training
            if add_noise and self.latent_noise_scale > 0:
                z = z + torch.randn_like(z) * self.latent_noise_scale

            return z

    def decode(
        self,
        z: torch.Tensor,
        mask: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """
        Decode latent representation to input space.

        Args:
            z: Latent representation (..., latent_dim)
            mask: Channel mask or indices for this session

        Returns:
            x_hat: Reconstructed input (..., C_s)
        """
        # Apply inverse flows
        h = self._apply_flow(z, reverse=True)

        # Project back to input space
        x_hat = self._project_up(h, mask)
        return x_hat

    def reconstruct(
        self,
        x: torch.Tensor,
        mask: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """Full reconstruction: encode then decode."""
        z = self.encode(x, mask, return_log_det=False)
        return self.decode(z, mask)

    def session_mats(
        self,
        mask: Union[torch.Tensor, list],
    ):
        """
        Get session-specific projection matrices (for compatibility).

        Note: For flow models, there's no simple matrix representation,
        so this returns the linear projection component only.
        """
        if self.w_proj is None:
            # No projection, return identity-like
            if isinstance(mask, list):
                idx = torch.tensor(mask, device=self.device, dtype=torch.long)
            else:
                idx = self._indices_from_mask(mask).to(self.device)

            dim = len(idx)
            w_enc = torch.eye(dim, device=self.device)
            w_dec = w_enc.t()
            return w_enc, w_dec, idx

        if isinstance(mask, list):
            idx = torch.tensor(mask, device=self.w_proj.device, dtype=torch.long)
        else:
            idx = self._indices_from_mask(mask).to(self.w_proj.device)

        w_sel = self.w_proj[:, idx]  # [latent_dim, C_s]
        w_enc_s = w_sel
        w_dec_s = w_sel.t().contiguous()  # [C_s, latent_dim]
        return w_enc_s, w_dec_s, idx

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mask_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE reconstruction loss over present channels.

        Args:
            x: Ground truth (..., C_s)
            x_hat: Reconstruction (..., C_s)
            mask_present: Optional bool mask (C_s,) for valid channels

        Returns:
            loss: Scalar reconstruction loss
        """
        if mask_present is not None:
            while mask_present.ndim < x.ndim:
                mask_present = mask_present.unsqueeze(0)
            diff2 = ((x - x_hat) ** 2) * mask_present
            denom = mask_present.sum().clamp_min(1.0)
            return diff2.sum() / denom
        else:
            return nn.functional.mse_loss(x_hat, x)

    def get_optim(
        self,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    ):
        """
        Get optimizer for training.

        Args:
            lr: Learning rate (default: 1e-4)
            betas: Adam betas (default: (0.9, 0.999))
            eps: Adam epsilon (default: 1e-8)
            weight_decay: L2 regularization strength (default: 1e-4, increased from 0.0)
            amsgrad: Use AMSGrad variant (default: False)
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


class SessionDispatcherFlowAE(SessionDispatcher):
    """Session dispatcher for FlowChannelAE (same interface as LinearAE)."""

    DISPATCH_METHODS = [
        "encode",
        "decode",
        "pca_warm_start",
        "reconstruct",
        "session_mats",
    ]

    def register_closure(self, name, values):
        for dm in SessionDispatcherFlowAE.DISPATCH_METHODS:
            sig = inspect.signature(getattr(FlowChannelAE, dm))
            if name in sig.parameters:
                self.close_kwarg(dm, name, values)

    def register_masks(self, masks: dict):
        return self.register_closure("mask", masks)


def dispatch_warm_start(aes, masks, data, is_identity=False, device=None):
    """Warm start autoencoders with PCA or identity."""
    for session_id in data.keys():
        if not isinstance(data, dict):
            d = rotate_session_from_batch(data, session_id, device=device)
        else:
            d = data[session_id]

        if is_identity:
            aes.instances[session_id].identity_warm_start()
        else:
            mask = masks[session_id]
            x = torch.cat((d[0], d[2]), dim=1)
            aes.instances[session_id].pca_warm_start(x, mask=mask)


def make_masks(data, device=None):
    """Create channel masks for each session."""
    masks = {}
    for session_id in data.session_ids:
        masks[session_id] = torch.arange(
            data.get_session_num_feats(session_id)
        ).to(device)
    return masks


def from_cfg_single(cfg, in_dim, device=None, **kwargs):
    """Create single autoencoder instance from config."""
    import hydra.utils
    ae = hydra.utils.instantiate(
        cfg.ae.module, in_dim=in_dim, device=device, **kwargs
    )
    return ae


def from_cfg_and_in_dims(cfg, in_dims: dict, shared=False, device=None, **kwargs):
    """
    Create autoencoders for multiple sessions.

    Args:
        cfg: Hydra config
        in_dims: {session_id: in_dim}
        shared: If True, use single shared model across sessions
        device: Device to place models on

    Returns:
        SessionDispatcher with autoencoder instances
    """
    import hydra.utils

    instances = {}

    if shared:
        in_dim = max(in_dims.values())
        instance = from_cfg_single(cfg, in_dim, device=device, **kwargs)
        for session_id in in_dims.keys():
            instances[session_id] = instance
    else:
        for session_id, in_dim in in_dims.items():
            instance = from_cfg_single(cfg, in_dim, device=device, **kwargs)
            instances[session_id] = instance

    return hydra.utils.instantiate(cfg.ae.dispatcher, instances)


def from_cfg_and_data(cfg, data, shared=False, warm_start=True, device=None, **kwargs):
    """
    Create autoencoders from config and data.

    Args:
        cfg: Hydra config
        data: MultiSession data object
        shared: If True, use single shared model
        warm_start: If True, initialize with PCA
        device: Device to place models on

    Returns:
        SessionDispatcher with initialized autoencoders
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
            aes,
            masks,
            data,
            device=device,
            is_identity=cfg.ae.warm_start_is_identity,
        )

    return aes
