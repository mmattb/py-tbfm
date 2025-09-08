"""
PyTorch implementation of the temporal basis function model.
"""

import inspect

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
        embed_dim_rest: int | None = None,
        embed_dim_stim: int | None = None,
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
            embed_dim_rest=embed_dim_rest,
            embed_dim_stim=embed_dim_stim,
            device=device,
        )

    def get_optim(self, lr=1e-4):
        """
        Get a PyTorch optimizer for training this model.
        Args:
            lr [float]: learning rate
        """
        ml = [self.basis_weighting, self.bases]
        ml = nn.ModuleList(ml)
        return torch.optim.AdamW(ml.parameters(), lr=lr)

    def get_weighting_reg(self):
        """
        Returns the Frobenius norm of the basis weightings, used to regularize during training.
        """
        # This assumes that we ought to weight each of these evenly?
        #  Seems okay.
        l = torch.linalg.norm(self.basis_weighting.weight, ord="fro")
        return l

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
        self.prev_bases = bases

        # basis_weights: (batch, in_dim * num_bases)
        basis_weights = self.basis_weighting(runway.flatten(start_dim=1))
        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))

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
