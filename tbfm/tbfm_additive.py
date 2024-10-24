"""
PyTorch implementation of the temporal basis function model with
forward stagewise additive modeling (FSAM).
"""

import torch
from torch import nn

from .bases import Bases
from . import utils


class TBFMAdditive(nn.Module):
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
        device []: something tensor.to() would accept
        """
        super().__init__()

        self.num_bases = num_bases
        self.stimdim = stimdim
        self.in_dim = in_dim
        self.trial_len = trial_len
        self.latent_dim = latent_dim
        self.basis_depth = basis_depth
        self.device = device

        # batchy: (batch, time, channel)
        # We are Z scoring with channels. So:
        flat = batchy.flatten(end_dim=1)
        self.mean = torch.mean(flat, axis=0).unsqueeze(0).to(device)
        self.std = torch.std(flat, axis=0).unsqueeze(0).to(device)
        assert self.mean.shape == self.std.shape == (1, batchy.shape[-1])

        self.basis_weighting = nn.Linear(runway * in_dim, num_bases * in_dim).to(device)
        self.basis_weighting.weight.retain_grad()
        self.basis_weighting.bias.retain_grad()

        self._basis = self._get_new_base()
        self._bases = []

        # The most recent set of bases materialized into a tensor
        self.bases = None

        # Index of basis we are currently training
        self.cbasis_idx = 0
        # Indexes all bases trained thus far or are presently training.
        #  For example range(5) if we have trained 4 and are working on 1 more.
        self.basis_selector = tuple(range(self.cbasis_idx + 1))

    def _get_new_base(self):
        """
        Make a new basis function generator network and return it.
        """
        basis = Bases(
            self.stimdim,
            1,
            self.trial_len,
            latent_dim=self.latent_dim,
            basis_depth=self.basis_depth,
            device=self.device,
        )
        return basis

    def get_optim(self, lr=1e-4):
        optim = torch.optim.AdamW(self.basis_weighting.parameters(), lr=lr)

        optimbp = torch.optim.AdamW(self._basis.parameters(), lr=lr)
        optim.param_groups.append(optimbp.param_groups[0])

        return optim

    def get_weighting_reg(self):
        """
        Returns the Frobenius norm of the basis weightings, used to regularize during training.
        """
        # This assumes that we ought to weight each of these evenly?
        #  Seems okay.
        w = self.basis_weighting.weight
        # wb = self.basis_weighting.bias
        l = torch.linalg.norm(w, ord="fro")
        # l = l + torch.linalg.norm( wb.unsqueeze(-1), ord="fro")

        return l

    def add_basis(self, optim):
        """
        Signal we are done with the basis we are presently training.
        Make a new one.
        Args:
            optim [torch.optim.Optimizer]: optimizer to which we will add the new basis
        """
        self._bases.append(self._basis)
        self._basis = self._get_new_base()

        # Add the new basis to the optimizer, replacing the prior basis
        optimbp = torch.optim.AdamW(
            self._basis.parameters(), lr=optim.param_groups[0]["lr"]
        )
        optim.param_groups[-1] = optimbp.param_groups[0]

        self.cbasis_idx += 1
        self.basis_selector = tuple(range(self.cbasis_idx + 1))

        self.basis_weighting.weight.retain_grad()
        self.basis_weighting.bias.retain_grad()

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
        runway = self.zscore(runway)
        x0 = runway[:, -1:, :]

        # bases: (all batch, time, num_bases)
        bases = []
        for basis in self._bases:
            bases.append(basis(stiminds))
        bases.append(self._basis(stiminds))
        bases = torch.cat(bases, axis=2)

        self.bases = bases

        # basis_weights: (batch, in_dim * num_bases)
        basis_weights = self.basis_weighting(runway.flatten(start_dim=1))
        # basis_weights: (batch, in_dim, num_bases)
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))
        basis_weights = basis_weights[:, :, self.basis_selector]

        # cpreds: (batch, time (after runway), in_dim))
        preds = (basis_weights @ bases.permute(0, 2, 1)).permute(0, 2, 1)

        # preds: (batch, time (after runway), all in_dim)
        preds = preds + x0

        return preds
