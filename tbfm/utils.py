"""
Reusable functions and math-y things go here.
Take no deps on other tbfm modules; this is a leaf.
"""

import torch
import torch.nn as nn


def zscore(data, mean, std):
    """
    data: (batch, time, channel)
    mean: (channel,)
    std: (channel,)
    """
    demeaned = data - mean
    zscored = demeaned / std
    return zscored


def zscore_inv(data, mean, std):
    """
    data: (batch, time, channel)
    mean: (channel,)
    std: (channel,)
    """
    unscaled = data * std
    un_zscored = unscaled + mean
    return un_zscored


def percentile_affine(x, p_low=0.1, p_high=0.9, floor=1e-3):
    # x: [B, T, C] few-shot stim samples
    flat = x.flatten(end_dim=1)
    q = torch.tensor([p_low, 0.5, p_high], device=x.device, dtype=x.dtype)
    qs = torch.quantile(flat, q, dim=0)  # [3, C]
    ql, qm, qh = qs[0], qs[1], qs[2]
    s = (qh - ql).clamp_min(floor * (qh - ql).median().clamp_min(1e-6))
    a = 2.0 / s
    b = -(qh + ql) / 2.0  # Average high and low; that's our new center
    # returns scale a and bias b such that x' = a*(x + b)
    return a, b


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
        self.register_buffer("var", None)

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        """
        x: [B, T, C]
        """
        flat = x.flatten(end_dim=1)
        self.mean = torch.mean(flat, axis=0).unsqueeze(0).to(x.device)
        self.std = torch.std(flat, axis=0).unsqueeze(0).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.var is None:
            raise RuntimeError("Call fit() before forward().")
        return zscore(x, self.mean, self.std)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.var is None:
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
        if self.mean is None or self.var is None:
            raise RuntimeError("Call fit() before forward().")
        return y / self.a - self.b
