import torch
from torch import nn

from utils import zscore, zscore_inv, percentile_affine


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
        if self.mean is None or self.var is None:
            raise RuntimeError("Call fit() before forward().")
        return y / self.a - self.b
