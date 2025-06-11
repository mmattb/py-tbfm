"""
PyTorch implementation of the phase invariant temporal basis function model.
"""

import torch
from torch import nn

from .bases import Bases
from . import utils


class TBFMPi(nn.Module):
    """
    Implements the phase invariant temporal basis function model
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
        self.stimdim = stimdim
        self.in_dim = in_dim
        self.device = device
        self.prev_bases = None

        self.should_zscore = zscore
        if zscore:
            self.set_z_params(batchy)

        # One set of basis weights *and phases* for each channel.
        # The addition of phases is what makes this distinct from vanilla TBFMs
        # The 2x multiplication is where we get phases. The design matrix will be
        # 2 blocks: weights | phases.
        self.basis_weights_and_phases = nn.Linear(runway * in_dim, 2 * num_bases * in_dim).to(device)

        self.bases = Bases(
            stimdim,
            num_bases,
            trial_len,
            latent_dim=latent_dim,
            basis_depth=basis_depth,
            device=device,
        )

    def set_z_params(self, batch):
        """
        Call to set/reset means and stdevs used for z scoring.
        """
        if not self.should_zscore:
            raise RuntimeError("We cannot adapt Zscoring if we aren't Zscoring!")

        # batchy: (batch, time, channel)
        # We are Z scoring with channels. So:
        flat = batch.flatten(end_dim=1)
        self.mean = torch.mean(flat, axis=0).unsqueeze(0).to(self.device)
        self.std = torch.std(flat, axis=0).unsqueeze(0).to(self.device)
        assert self.mean.shape == self.std.shape == (1, batch.shape[-1])

    def get_optim(self, lr=1e-4):
        """
        Get a PyTorch optimizer for training this model.
        Args:
            lr [float]: learning rate
        """
        ml = nn.ModuleList([self.basis_weights_and_phases, self.bases])
        return torch.optim.AdamW(ml.parameters(), lr=lr)

    def get_weighting_reg(self):
        """
        Returns the Frobenius norm of the basis weightings, used to regularize during training.
        """
        # This assumes that we ought to weight each of these evenly?
        #  Seems okay.
        output_size = self.basis_weights_and_phases.weight.shape[0]
        cut = output_size // (self.in_dim * self.num_bases)
        # TODO: for now let's only regularize the weights, not the phases.
        l = torch.linalg.norm(self.basis_weights_and_phases.weight[:cut, :], ord="fro")
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
        return utils.zscore(data, self.mean, self.std)

    def forward(self, runway, stiminds):
        """
        Args:
            runway [tensor]: (batch_size, runway length, in_dim)
            stiminds [tensor]: (batch_size, trial_len, stimdim)
        Returns:
            y_hat [tensor]: (batch_size, trial_len, in_dim)
        """
        if self.should_zscore:
            runway = self.zscore(runway)

        # runway: (batch, time (runway), in_dim)
        # stiminds: (batch, time (after runway), stimdim)
        x0 = runway[:, -1:, :]


        # bases: (all batch, time, num_bases)
        # basis_weights: (batch, in_dim * num_bases * 2)
        basis_weights_and_phases = self.basis_weights_and_phases(runway.flatten(start_dim=1))
        num_weights = self.in_dim * self.num_bases
        basis_weights = basis_weights_and_phases[:, :num_weights]
        basis_phases = basis_weights_and_phases[:, num_weights:]

        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))

        # basis_phases: (batch, in_dim, num_bases)
        basis_phases = basis_phases.unflatten(1, (self.in_dim, self.num_bases))

        # bases: (all batch, time, num_bases)
        bases = self.bases(stiminds)
        self.prev_bases = bases
        # bases: (all batch, num_bases, time)
        bases = bases.permute(0, 2, 1)
        # (all batch, in_dim, num_bases, time)
        bases = utils.fft_circular_shift_per_basis(bases, basis_phases)

        # preds: (batch, time (after runway), in_dim)
        preds = (basis_weights @ bases).permute(0, 2, 1)
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
        self.mean = tbfm.mean
        self.std = tbfm.std
        self.basis_weights_and_phases = tbfm.basis_weights_and_phases
        self.in_dim = tbfm.in_dim
        self.num_bases = tbfm.num_bases

    def zscore(self, data):
        """
        z score some tensor of data using the means and stdevs this model learned. Handy for
        graphing and loss computation so it's exposed publicly.
        Args:
            data: tensor([batch_size, trial_len, in_dim])
        """
        return utils.zscore(data, self.mean, self.std)

    def forward(self, runway):
        """
        Args:
            runway [tensor]: (batch_size, runway length, in_dim)
        Returns:
            y_hat [tensor]: (batch_size, trial_len, in_dim)
        """
        runway = self.zscore(runway)

        # runway: (batch, time (runway), in_dim)
        x0 = runway[:, -1:, :]

        # bases: (all batch, time, num_bases)
        # basis_weights: (batch, in_dim * num_bases * 2)
        basis_weights_and_phases = self.basis_weights_and_phases(runway.flatten(start_dim=1))
        num_weights = self.in_dim * self.num_bases
        basis_weights = basis_weights_and_phases[:, :num_weights]
        basis_phases = basis_weights_and_phases[:, num_weights:]

        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))

        # basis_phases: (batch, in_dim, num_bases)
        basis_phases = basis_phases.unflatten(1, (self.in_dim, self.num_bases))

        # bases: (all batch, num_bases, time)
        bases = self.bases.permute(0, 2, 1)
        bases = utils.fft_circular_shift_per_basis(bases, basis_phases)

        # preds: (batch, time (after runway), in_dim)
        preds = (basis_weights @ bases).permute(0, 2, 1)
        preds = preds + x0

        return preds
