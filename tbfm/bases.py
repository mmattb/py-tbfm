"""
PyTorch Implementation of the basis generator network.
"""

import torch
from torch import nn


class Bases(nn.Module):
    """
    Basis generator network.
    """

    def __init__(
        self, stimdim, num_bases, trial_len, latent_dim=20, basis_depth=2, device=None
    ):
        """
        Args:
            stimdim [int]: dimensionality of the stimulation descriptor
            num_bases [int]: number of bases we will generate
            trial_len [int]: the number of time steps long each basis will be
            latent_dim [int]: width of hidden layers
            basis_depth [int]: number of hidden layers
            device []: something tensor.to() would accept
        """
        super().__init__()

        # Each basis maps stimdim -> 1
        self.num_bases = num_bases
        self.trial_len = trial_len

        in_dim = stimdim * trial_len
        ldim = latent_dim
        seq = [nn.Linear(in_dim, ldim), nn.Tanh()]
        for _ in range(basis_depth):
            seq.append(nn.Linear(ldim, ldim))
            seq.append(nn.Tanh())
        seq.append(nn.Linear(ldim, trial_len * num_bases))

        self._inner = nn.Sequential(*seq).to(device)

    def get_reg_weights(self):
        """
        Returns a regularizer value for this network
        """
        l = 0
        for layer in self._inner[:-1]:
            try:
                l += torch.linalg.norm(layer.weight)
            except AttributeError:
                pass
        return l

    def forward(self, stiminds):
        """
        stiminds: tensor([batch_size, trial_len, stimdim])
        """
        # stiminds: (batch, time, stimdim)
        stiminds = stiminds.flatten(start_dim=1)
        # stiminds: (batch, time*stimdim)
        bases = self._inner(stiminds)
        # bases: (batch, time*num_bases)
        bases = bases.unflatten(dim=1, sizes=(self.trial_len, self.num_bases))
        # bases: (batch, time, num_bases)
        return bases
