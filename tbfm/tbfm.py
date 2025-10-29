"""
PyTorch implementation of the temporal basis function model.
"""

import hydra.utils
import torch
from torch import nn

from .bases import Bases
from . import normalizers
from . import utils
from .utils import SessionDispatcher


class TBFM(nn.Module):
    """
    Implements the temporal basis function model
    """

    def __init__(
        self,
        in_dim,
        stimdim,
        runway,
        num_bases,
        trial_len,
        batchy,
        latent_dim=20,
        covariate_dim=None,
        basis_depth=2,
        zscore=True,
        use_film_bases: bool = False,
        proj_film_dim: int = 32,
        embed_dim_rest: int | None = None,
        embed_dim_stim: int | None = None,
        basis_gen_dropout: float = 0.0,
        device=None,
    ):
        """
        in_dim [int]: dimensionality of the time series, e.g. the number of electrodes
        stimdim [int]: dimenstionality of the stimulation descriptor
        runway [int]: length of the runway in time steps. That is: how many time steps' of data will be our x values?
        num_bases [int]: number of bases to learn
        trial_len [int]: length of our y values in time steps. i.e. the forecast horizon.
        batchy [int]: a batch of data used to calculate means and stdevs to z score later. Should be (batch_size, time, in_dim)  (any length of time)
        latent_dim [int]: width of hidden layers
        basis_depth [int]: number of hidden layers
        zscore [bool]: True if we should z score x values as they arrive, False otherwise
        device []: something tensor.to() would accept
        """
        super().__init__()

        self.num_bases = num_bases
        self.stimdim = stimdim or covariate_dim
        self.in_dim = in_dim
        self.device = device
        self.prev_bases = None
        self.prev_basis_weights = None
        self.use_film_bases = use_film_bases

        if zscore:
            self.normalizer = normalizers.ScalerZscore()
            self.normalizer.fit(batchy)
        else:
            self.normalizer = None

        # One set of basis weights for each channel.
        self.basis_weighting = nn.Linear(runway * in_dim, num_bases * in_dim).to(device)

        self.bases = Bases(
            self.stimdim,
            num_bases,
            trial_len,
            latent_dim=latent_dim,
            basis_depth=basis_depth,
            use_film=use_film_bases,
            proj_film_dim=proj_film_dim,
            embed_dim_rest=embed_dim_rest,
            embed_dim_stim=embed_dim_stim,
            basis_gen_dropout=basis_gen_dropout,
            device=device,
        )

    def reset_state(self):
        """
        Clear any cached state from previous forward passes.
        Call this before inner optimization loops to prevent graph retention issues.
        """
        self.prev_bases = None
        self.prev_basis_weights = None
        if hasattr(self.bases, "reset_state"):
            self.bases.reset_state()

    def get_optim(self, lr=1e-4):
        """
        Get a PyTorch optimizer for training this model.
        Args:
            lr [float]: learning rate
        """
        ml = [self.basis_weighting, self.bases]
        ml = nn.ModuleList(ml)
        return torch.optim.AdamW(ml.parameters(), lr=lr)

    def get_optim_custom_base(
        self,
        lr_head: float = 1e-3,
        lr_film: float = 3e-3,
        wd_head: float = 1e-4,
        wd_film: float = 1e-5,
    ):
        """
        Returns two optimizers:
          - optim_head: basis weighting + core Bases MLP (non-FiLM)
          - optim_film: FiLM-specific params (proj_rest, proj_stim, gate, film_head)
        """
        optim_bw = torch.optim.AdamW(
            self.basis_weighting.parameters(),
            lr=0.5 * lr_head,
            weight_decay=wd_head,
        )
        bases_core_params, bases_film_params = self.bases.get_params()

        # flatten
        bg_params = [p for group in bases_core_params for p in group]

        optim_bg = torch.optim.AdamW(bg_params, lr=lr_head, weight_decay=wd_head)
        optim_film = None
        if bases_film_params:
            optim_film = torch.optim.AdamW(
                bases_film_params, lr=lr_film, weight_decay=wd_film
            )

        return optim_bw, optim_bg, optim_film

    def get_weighting_reg(self):
        """
        Returns the Frobenius norm of the basis weightings, used to regularize during training.
        """
        W = self.basis_weighting.weight
        l = torch.linalg.norm(W, ord="fro") / (W.numel() ** 0.5)
        return l

    def get_basis_rms_reg(self, eps: float = 1e-8) -> torch.Tensor:
        """
        Softly encourages bases to be orthonormal over the time axis.
        eps : numerical stability.

        Returns
        -------
        loss : scalar tensor
        """
        # diag_weight : weight for pushing per-basis RMS toward 1 (||b_k||_2 over time).
        # offdiag_weight : weight for pushing cross-basis correlations toward 0.
        diag_weight: float = 1.0
        offdiag_weight: float = 1.0

        B = self.prev_bases
        bsz, T, K = B.shape

        # Gram over time: G[b] = (B[b]^T @ B[b]) / T, shape (b, k, k)
        G = torch.einsum("btk,btj->bkj", B, B) / (T + eps)

        # Diagonal toward 1 (unit RMS), off-diagonals toward 0 (orthogonality)
        diag = torch.diagonal(G, dim1=-2, dim2=-1)  # (b, k)
        loss_diag = (diag - 1.0).pow(2).mean()

        off = G - torch.diag_embed(diag)  # zero out diagonal
        loss_off = (off.pow(2)).mean()

        return diag_weight * loss_diag + offdiag_weight * loss_off

    def tbfm_compile(self, stiminds):
        """
        Return a compiled version of this TBFM.
        The compiled version operates the same but can no longer be trained.
        """
        if self.use_film_bases:
            raise NotImplementedError()
        return TBFMCompiled(self, stiminds)

    def zscore(self, data):
        """
        z score some tensor of data using the means and stdevs this model learned. Handy for
        graphing and loss computation so it's exposed publicly.
        Args:
            data: tensor([batch_size, trial_len, in_dim])
        """
        normalizer = self.normalizer
        if not isinstance(normalizer, normalizers.ScalerZscore):
            raise TypeError(
                f"Requested Z-score normalization but normalizer is set to {type(normalizer)}"
            )
        return normalizer(data)

    def normalize(self, data):
        if self.normalizer:
            data = self.normalizer(data)
        return data

    def forward(
        self,
        runway,
        stiminds,
        embedding_rest: torch.Tensor | None = None,
        embedding_stim: torch.Tensor | None = None,
    ):
        """
        Args:
            runway [tensor]: (batch_size, runway length, in_dim)
            stiminds [tensor]: (batch_size, trial_len, stimdim)
        Returns:
            y_hat [tensor]: (batch_size, trial_len, in_dim)
        """
        runway = self.normalize(runway)

        # runway: (batch, time (runway), in_dim)
        # stiminds: (batch, time (after runway), stimdim)
        x0 = runway[:, -1:, :]

        # bases: (all batch, time, num_bases)
        bases = self.bases(
            stiminds, embedding_rest=embedding_rest, embedding_stim=embedding_stim
        )
        # Store for regularization (keep graph for outer loop backprop)
        self.prev_bases = bases

        # basis_weights: (batch, in_dim * num_bases)
        basis_weights = self.basis_weighting(runway.flatten(start_dim=1))
        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))
        # XXX
        basis_weights = torch.tanh(basis_weights)
        basis_weights = torch.nn.functional.normalize(basis_weights, p=2, dim=-1)

        # Store for regularization (keep graph for outer loop backprop)
        self.prev_basis_weights = basis_weights

        # cpreds: (batch, time (after runway), in_dim)
        preds = (basis_weights @ bases.permute(0, 2, 1)).permute(0, 2, 1)

        # preds: (batch, time (after runway), in_dim)
        preds = preds + x0

        return preds


class TBFMCompiled(nn.Module):
    """
    A TBFM compiled from another TBFM
    """

    def __init__(self, tbfm, stiminds):
        """
        Args:
            tbfm [TBFM]: the TBFM we are compiling from
            stiminds [tensor]: the stimulation descriptor which the compiled bases will correspond to
        """
        super().__init__()
        self.device = tbfm.device
        self.bases = tbfm.bases(stiminds)
        self.normalizer = tbfm.normalizer
        self.basis_weighting = tbfm.basis_weighting
        self.in_dim = tbfm.in_dim
        self.num_bases = tbfm.num_bases

    def zscore(self, data):
        """
        z score some tensor of data using the means and stdevs this model learned. Handy for
        graphing and loss computation so it's exposed publicly.
        Args:
            data: tensor([batch_size, trial_len, in_dim])
        """
        normalizer = self.normalizer
        if not isinstance(normalizer, normalizers.ScalerZscore):
            raise TypeError(
                f"Requested Z-score normalization but normalizer is set to {type(normalizer)}"
            )
        return normalizer(data)

    def normalize(self, data):
        if self.normalizer:
            data = self.normalizer(data)
        return data

    def forward(self, runway):
        """
        Args:
            runway [tensor]: (batch_size, runway length, in_dim)
        Returns:
            y_hat [tensor]: (batch_size, trial_len, in_dim)
        """
        runway = self.normalize(runway)

        # runway: (batch, time (runway), in_dim)
        x0 = runway[:, -1:, :]

        # bases: (all batch, time, num_bases)
        # basis_weights: (batch, in_dim * num_bases)
        basis_weights = self.basis_weighting(runway.flatten(start_dim=1))
        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))

        # preds: (batch, time (after runway), in_dim)
        preds = (basis_weights @ self.bases.permute(0, 2, 1)).permute(0, 2, 1)
        preds = preds + x0

        return preds


class SessionDispatcherTBFM(SessionDispatcher):
    DISPATCH_METHODS = [
        "zscore",
        "normalize",
    ]

    @property
    def single(self):
        fk = list(self.instances.keys())[0]
        return self.instances[fk]

    @property
    def bases(self):
        return self.single.bases

    def __call__(self, runways, covariates, embeddings_rest=None, embeddings_stim=None):
        # module-style call override
        y_hats = {}

        ck = self.get_closed_kwargs("__call__")

        for sid, runway in runways.items():
            _covariates = covariates[sid]

            instance = self.instances[sid]

            if embeddings_rest:
                embedding_rest = embeddings_rest[sid]
            elif sid in ck and "embeddings_rest" in ck[sid]:
                embedding_rest = ck[sid]["embeddings_rest"]
            else:
                embedding_rest = None

            if embeddings_stim:
                embedding_stim = embeddings_stim[sid]
            elif sid in ck and "embeddings_stim" in ck[sid]:
                embedding_stim = ck[sid]["embeddings_stim"]
            else:
                embedding_stim = None

            y_hat = instance(
                runway,
                _covariates,
                embedding_rest=embedding_rest,
                embedding_stim=embedding_stim,
            )
            y_hats[sid] = y_hat
        return y_hats

    def register_embeds_film(self, embeddings_rest, embeddings_stim):
        self.close_kwarg("__call__", "embeddings_rest", embeddings_rest)
        self.close_kwarg("__call__", "embeddings_stim", embeddings_stim)


def shared_from_cfg_and_base(cfg, session_ids, base_path, **kwargs):
    # stimdim=None triggers use of covariate_dim instead; this is a backwards compat thing
    instance = hydra.utils.instantiate(
        cfg.tbfm.module, stimdim=None, zscore=False, batchy=None, **kwargs
    )

    # Load base model weights.
    # Note the base must have the same cfg - e.g. latent dim
    instance.load_state_dict(torch.load(base_path, weights_only=True))
    instance.eval()

    instances = {}  # {session_id, instance}
    for session_id in session_ids:
        instances[session_id] = instance
    return hydra.utils.instantiate(cfg.tbfm.dispatcher, instances)


def from_cfg(cfg, session_ids, shared=False, **kwargs):
    instances = {}  # {session_id, instance}

    if shared:
        # stimdim=None triggers use of covariate_dim instead; this is a backwards compat thing
        instance = hydra.utils.instantiate(
            cfg.tbfm.module, stimdim=None, zscore=False, batchy=None, **kwargs
        )
        for session_id in session_ids:
            instances[session_id] = instance
    else:
        # stimdim=None triggers use of covariate_dim instead; this is a backwards compat thing
        for session_id in session_ids:
            instance = hydra.utils.instantiate(
                cfg.tbfm.module, stimdim=None, zscore=False, batchy=None, **kwargs
            )
            instances[session_id] = instance
    return hydra.utils.instantiate(cfg.tbfm.dispatcher, instances)
