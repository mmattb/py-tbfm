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


class FourierBases(nn.Module):
    """
    Fourier basis generator network.
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
        seq = [nn.Linear(in_dim, ldim), nn.Sigmoid()]
        for _ in range(basis_depth):
            seq.append(nn.Linear(ldim, ldim))
            seq.append(nn.Sigmoid())
        self.num_freqs = trial_len // 2 + 1  # number of unique frequencies in FFT
        seq.append(nn.Linear(ldim, self.num_freqs * num_bases * 2))  # real + imag parts

        self._inner = nn.Sequential(*seq).to(device)

    def forward(self, stiminds):
        """
        stiminds: tensor([batch_size, trial_len, stimdim])
        Returns spectrum with Hermitian symmetry to ensure real-valued output.
        """
        batch_size = stiminds.shape[0]
        # stiminds: (batch, time, stimdim)
        stiminds = stiminds.flatten(start_dim=1)
        # stiminds: (batch, time*stimdim)
        out = self._inner(stiminds)  # (batch_size, num_freqs * num_bases * 2)
        out = out.view(
            batch_size, -1, self.num_freqs, 2
        )  # (batch_size, num_bases, num_freqs, 2)
        spectrum = torch.complex(
            out[..., 0], out[..., 1]
        )  # (batch_size, num_bases, num_freqs)

        # Force DC component (0 freq) to be real
        spectrum[..., 0] = spectrum[..., 0].real

        # For even-length signals, Nyquist frequency should be real
        if self.trial_len % 2 == 0:
            spectrum[..., -1] = spectrum[..., -1].real

        # Return with shape (batch_size, num_freqs, num_bases)
        return spectrum.permute(0, 2, 1)
