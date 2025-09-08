from hydra.utils import instantiate
import torch
from torch import nn

from .utils import (
    zscore,
    zscore_inv,
    percentile_affine,
    SessionDispatcher,
    rotate_session_from_batch,
)


class SessionDispatcherScaler(SessionDispatcher):
    DISPATCH_METHODS = ["fit", "forward", "inverse"]


def from_cfg_single(cfg, data, device=None):
    x = torch.cat((data[0], data[2]), dim=1)  # 20, 164 -> 184
    normalizer = instantiate(cfg.normalizers.module)
    normalizer.fit(x)
    normalizer = normalizer.to(device)
    return normalizer


def from_cfg(cfg, data, device=None):
    instances = {}  # {session_id, instance}

    for session_id in data.keys():
        if not isinstance(data, dict):
            d = rotate_session_from_batch(data, session_id, device=device)
        else:
            d = data[session_id]
        instance = from_cfg_single(cfg, d, device=device)
        instances[session_id] = instance
    return SessionDispatcherScaler(instances)


class ScalerZscore(nn.Module):
    """
    Per-channel Z-scorer
    - fit(x): x shape [B, T, C]; computes per-channel mean/var
    - forward(x): normalize/zscore the data
    - inverse(Y): denormalize
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("std", None)

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        """
        x: [B, T, C]
        """
        flat = x.flatten(end_dim=1)
        self.mean = torch.mean(flat, axis=0).unsqueeze(0).to(x.device)
        self.std = torch.std(flat, axis=0).unsqueeze(0).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() before forward().")
        return zscore(x, self.mean, self.std)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() before forward().")
        return zscore_inv(y, self.mean, self.std)


class ScalerQuant(nn.Module):
    """
    Per-channel quantile-based normalizer.
    Similar to z-scoring but more robust to extreme values, making it
    maybe more robust to estimate.
    - fit(x): x shape [B, T, C]; computes per-channel statistics
    - forward(x): normalize the data
    - inverse(Y): denormalize
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("a", None)
        self.register_buffer("b", None)

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        """
        x: [B, T, C]
        """
        # returns scale a and bias b such that x' = a*(x + b)
        a, b = percentile_affine(x)
        self.a = a.to(x.device)
        self.b = b.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a is None or self.b is None:
            raise RuntimeError("Call fit() before forward().")
        return (x + self.b) * self.a

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.a is None or self.b is None:
            raise RuntimeError("Call fit() before inverse().")
        return y / self.a - self.b
