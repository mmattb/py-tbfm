import inspect
import hydra.utils
import torch
import torch.nn as nn
from typing import Optional, Union, Dict
import numpy as np
import scipy.spatial.distance

from .utils import SessionDispatcher, rotate_session_from_batch


def build_spatial_structures(node_position: Dict[int, tuple], mask_indices: Optional[list] = None):
    """
    Build spatial structures from electrode positions.

    Args:
        node_position: dict mapping node_id -> (x, y) coordinates
        mask_indices: optional list of indices to subset (for sessions with missing channels)

    Returns:
        distance_matrix: (C, C) pairwise Euclidean distances
        adjacency_matrix: (C, C) weighted adjacency (based on Gaussian kernel of distances)
        spatial_coords: (C, 2) coordinate array
    """
    # Sort by node_id to ensure consistent ordering
    sorted_ids = sorted(node_position.keys())

    # If mask provided, filter to those indices
    if mask_indices is not None:
        sorted_ids = [sid for sid in sorted_ids if sid in mask_indices]

    # Extract coordinates
    spatial_coords = np.array([node_position[nid] for nid in sorted_ids])  # (C, 2)

    # Compute pairwise Euclidean distance matrix
    distance_matrix = scipy.spatial.distance.cdist(spatial_coords, spatial_coords, metric='euclidean')

    # Build weighted adjacency using Gaussian kernel
    # sigma is set to median distance to capture local structure
    sigma = np.median(distance_matrix[distance_matrix > 0])
    adjacency_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
    np.fill_diagonal(adjacency_matrix, 0)  # No self-connections

    return distance_matrix, adjacency_matrix, spatial_coords


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
        use_spatial: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.use_lora = use_lora
        self.device = device
        self.use_spatial = use_spatial

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

        # Rank-1 LoRA (global directions); per-session you'll scale with a scalar alpha if you want
        if use_lora:
            self.a = nn.Parameter(torch.zeros(latent_dim).to(device))  # left direction
            self.b = nn.Parameter(torch.zeros(in_dim).to(device))  # right direction
            # init tiny so base dominates
            nn.init.normal_(self.a, std=1e-3)
            nn.init.normal_(self.b, std=1e-3)

        # Spatial structures (registered buffers, not learnable parameters)
        if use_spatial:
            self.register_buffer('adjacency_matrix', None)
            self.register_buffer('distance_matrix', None)
            self.register_buffer('laplacian_matrix', None)

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

    def register_spatial_structure(
        self,
        node_position: Dict[int, tuple],
        mask_indices: Optional[list] = None
    ):
        """
        Register spatial structure from electrode positions.

        Args:
            node_position: dict mapping node_id -> (x, y) coordinates
            mask_indices: optional list of indices for present channels
        """
        if not self.use_spatial:
            raise ValueError("use_spatial=False, cannot register spatial structure")

        # Build spatial structures
        dist_mat, adj_mat, coords = build_spatial_structures(node_position, mask_indices)

        # Compute graph Laplacian: L = D - A
        # where D is degree matrix (diagonal with row sums of A)
        degree = np.sum(adj_mat, axis=1)
        laplacian = np.diag(degree) - adj_mat

        # Convert to torch tensors and register as buffers
        self.adjacency_matrix = torch.from_numpy(adj_mat).float().to(self.device)
        self.distance_matrix = torch.from_numpy(dist_mat).float().to(self.device)
        self.laplacian_matrix = torch.from_numpy(laplacian).float().to(self.device)

    def spatial_smoothness_penalty(
        self,
        x: torch.Tensor,
        laplacian: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute spatial smoothness penalty using graph Laplacian.
        Penalizes differences between spatially adjacent electrodes.

        Args:
            x: (..., C) tensor of channel activations
            laplacian: (C, C) graph Laplacian matrix. If None, uses self.laplacian_matrix
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Smoothness penalty scalar or tensor

        The penalty is: x^T L x = sum_ij L_ij * x_i * x_j
        which penalizes large differences between connected nodes.
        """
        if laplacian is None:
            if self.laplacian_matrix is None:
                raise ValueError("No Laplacian registered. Call register_spatial_structure first.")
            laplacian = self.laplacian_matrix

        # x: (..., C), L: (C, C)
        # x^T L x for each sample
        # Optimized: avoid einsum, use matrix multiplication
        # xLx = sum((x @ L) * x, dim=-1)
        xL = torch.matmul(x, laplacian)  # (..., C)
        xLx = (xL * x).sum(dim=-1)  # (...,)

        if reduction == 'mean':
            return xLx.mean()
        elif reduction == 'sum':
            return xLx.sum()
        else:
            return xLx

    def spatial_decoder_penalty(
        self,
        mask: Union[torch.Tensor, list],
        lora_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Penalize decoder weights to encourage spatial smoothness.
        Each decoder column should be spatially smooth (nearby electrodes have similar decoder weights).

        Returns:
            Penalty encouraging spatial smoothness in decoder weights
        """
        if self.laplacian_matrix is None:
            raise ValueError("No Laplacian registered. Call register_spatial_structure first.")

        _, w_dec_s, _ = self.session_mats(mask, lora_alpha)  # (C_s, latent_dim)

        # For each latent dimension, penalize spatial roughness of decoder weights
        # w_dec_s^T L w_dec_s, where rows are latent dims
        # Optimized: compute L @ w_dec_s first, then use element-wise operations
        Lw = torch.matmul(self.laplacian_matrix, w_dec_s)  # (C_s, latent_dim)
        penalty = (w_dec_s * Lw).sum()  # sum of element-wise products = trace(W^T L W)

        return penalty / (self.latent_dim * w_dec_s.shape[0])  # normalize

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
