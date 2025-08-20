"""
PyTorch implementation of the temporal basis function model.
"""

import torch
from torch import nn

from .bases import Bases
from . import utils


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
        basis_depth=2,
        zscore=True,
        normalizer=None,
        ae=None,
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
        ae [nn.Module]: an optional autoencoder with encode() and decode() methods.
                        When present, the TBFM lives in the latent space. We assume in that
                        case that in_dim is equal to the encoder's latent dimension.
        device []: something tensor.to() would accept
        """
        super().__init__()

        self.num_bases = num_bases
        self.stimdim = stimdim
        self.in_dim = in_dim
        self.device = device
        self.prev_bases = None
        self.ae = ae

        self.should_normalize = zscore or normalizer is not None
        if zscore:
            self.normalizer = utils.ScalerZscore()
            self.normalizer.fit(batchy)
        elif normalizer is not None:
            if type(normalizer) is type:
                self.normalizer = normalizer()
                self.normalizer.fit(batchy)
            else:
                self.normalizer = normalizer

        # One set of basis weights for each channel.
        self.basis_weighting = nn.Linear(runway * in_dim, num_bases * in_dim).to(device)

        self.bases = Bases(
            stimdim,
            num_bases,
            trial_len,
            latent_dim=latent_dim,
            basis_depth=basis_depth,
            device=device,
        )

    def get_optim(self, lr=1e-4, include_ae=False):
        """
        Get a PyTorch optimizer for training this model.
        Args:
            lr [float]: learning rate
        """
        ml = [self.basis_weighting, self.bases]
        if include_ae:
            ml.append(self.ae)
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
        return TBFMCompiled(self, stiminds)

    def zscore(self, data):
        """
        z score some tensor of data using the means and stdevs this model learned. Handy for
        graphing and loss computation so it's exposed publicly.
        Args:
            data: tensor([batch_size, trial_len, in_dim])
        """
        normalizer = self.normalizer
        if not isinstance(normalizer, utils.ScalerZscore):
            raise TypeError(
                f"Requested Z-score normalization but normalizer is set to {type(normalizer)}"
            )
        return normalizer(data)

    def normalize(self, data):
        return self.normalizer(data)

    def encode(self, data):
        if self.ae is not None:
            return self.ae.encode(data)
        return data

    def decode(self, data):
        if self.ae is not None:
            return self.ae.decode(data)
        return data

    def forward(self, runway, stiminds):
        """
        Args:
            runway [tensor]: (batch_size, runway length, in_dim)
            stiminds [tensor]: (batch_size, trial_len, stimdim)
        Returns:
            y_hat [tensor]: (batch_size, trial_len, in_dim)
        """
        if self.should_normalize:
            runway = self.normalize(runway)
        runway = self.encode(runway)

        # runway: (batch, time (runway), in_dim)
        # stiminds: (batch, time (after runway), stimdim)
        x0 = runway[:, -1:, :]

        # bases: (all batch, time, num_bases)
        bases = self.bases(stiminds)
        self.prev_bases = bases

        # basis_weights: (batch, in_dim * num_bases)
        basis_weights = self.basis_weighting(runway.flatten(start_dim=1))
        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))

        # cpreds: (batch, time (after runway), in_dim)
        preds = (basis_weights @ bases.permute(0, 2, 1)).permute(0, 2, 1)

        # preds: (batch, time (after runway), in_dim)
        preds = preds + x0

        preds = self.decode(preds)

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
        self.ae = tbfm.ae
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
        if not isinstance(normalizer, utils.ScalerZscore):
            raise TypeError(
                f"Requested Z-score normalization but normalizer is set to {type(normalizer)}"
            )
        return normalizer(data)

    def normalize(self, data):
        return self.normalizer(data)

    def encode(self, data):
        if self.ae is not None:
            return self.ae.encode(data)
        return data

    def decode(self, data):
        if self.ae is not None:
            return self.ae.decode(data)
        return data

    def forward(self, runway):
        """
        Args:
            runway [tensor]: (batch_size, runway length, in_dim)
        Returns:
            y_hat [tensor]: (batch_size, trial_len, in_dim)
        """
        runway = self.zscore(runway)
        runway = self.encode(runway)

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

        preds = self.decode(preds)

        return preds
