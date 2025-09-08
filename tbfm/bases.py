"""
PyTorch Implementation of the basis generator network.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Bases(nn.Module):
    """
    Basis generator network with optional FiLM modulation.
    FiLM is applied after the input layer (first hidden activations).
    """

    def __init__(
        self,
        stimdim: int,
        num_bases: int,
        trial_len: int,
        latent_dim: int = 20,
        basis_depth: int = 2,
        use_film: bool = False,
        embed_dim_rest: int | None = None,
        embed_dim_stim: int | None = None,
        device=None,
    ):
        """
        Args:
            stimdim [int]: dimensionality of the stimulation descriptor
            num_bases [int]: number of bases we will generate
            trial_len [int]: the number of time steps long each basis will be
            latent_dim [int]: width of hidden layers
            basis_depth [int]: number of hidden layers
            use_film: 'tis a bool, my love
            embed_dim_rest: dimensionality of the rest data embedding, for FiLM
            embed_dim_stim: dimensionality of the stim data embedding, for FiLM
            device []: something tensor.to() would accept
        """
        super().__init__()

        self.num_bases = num_bases
        self.trial_len = trial_len
        self.latent_dim = latent_dim
        self.use_film = use_film

        in_dim = stimdim * trial_len

        # Core MLP
        self.in_layer = nn.Linear(in_dim, latent_dim)
        self.in_activation = nn.Tanh()

        self.hiddens = nn.ModuleList()
        for _ in range(basis_depth):
            self.hiddens.append(nn.Linear(latent_dim, latent_dim))
        self.hidden_activation = nn.Tanh()

        self.out_layer = nn.Linear(latent_dim, trial_len * num_bases)

        # ---- FiLM modulator (GELU in modulator paths) ----
        self.embed_dim_stim = embed_dim_stim
        self.embed_dim_rest = embed_dim_rest
        if use_film:
            mod_width = max(32, latent_dim)

            # rest and stim projection heads (GELU)
            self.proj_rest = nn.Sequential(
                nn.Linear(embed_dim_rest, mod_width),
                nn.GELU(),
            )
            if embed_dim_stim is not None:
                self.proj_stim = nn.Sequential(
                    nn.Linear(embed_dim_stim, mod_width),
                    nn.GELU(),
                )
                # conservative scalar gate in [0,1] driven by rest
                self.gate_rest_to_stim = nn.Sequential(
                    nn.Linear(embed_dim_rest, 1),
                    nn.Sigmoid(),
                )
            else:
                self.proj_stim = None
                self.gate_rest_to_stim = None

            # FiLM head → (gamma | beta) for first hidden width
            self.film_head = nn.Sequential(
                nn.Linear(mod_width, mod_width),
                nn.GELU(),
                nn.Linear(mod_width, 2 * latent_dim),
            )
            # init near identity: gamma ≈ 0, beta ≈ 0 → (1+gamma) * h + beta ≈ h
            nn.init.zeros_(self.film_head[-1].weight)
            nn.init.zeros_(self.film_head[-1].bias)

        self.to(device)

    def fuse_embeddings(
        self, embedding_rest: torch.Tensor, embedding_stim: torch.Tensor | None
    ):
        """
        Fuse rest and stim embeddings conservatively:
          fused = rest + gate(rest)*stim.
        embedding_rest: (batch_size, self.embed_dim_rest)
        embedding_stim: (batch_size, self.embed_dim_stim)
        """
        rest_proj = self.proj_rest(embedding_rest)
        if (embedding_stim is not None) and (self.proj_stim is not None):
            stim_proj = self.proj_stim(embedding_stim)
            gate = self.gate_rest_to_stim(embedding_rest)  # (batch, 1)
            return rest_proj + gate * stim_proj
        return rest_proj

    def get_reg_weights(self):
        total = 0.0
        for layer in [self.in_layer, *self.hiddens]:
            total = total + torch.linalg.norm(layer.weight)
        return total

    def film_reg(
        self, embedding_rest: torch.Tensor, embedding_stim: torch.Tensor | None = None
    ):
        """Small L2 on FiLM outputs to bias toward identity modulation."""
        if not self.use_film:
            return torch.tensor(0.0, device=embedding_rest.device)
        fused_embed = self.fuse_embeddings(embedding_rest, embedding_stim)
        gamma_beta = self.film_head(fused_embed)  # (batch, 2*width)
        half = gamma_beta.size(-1) // 2
        gamma = gamma_beta[:, :half]
        beta = gamma_beta[:, half:]
        return gamma.pow(2).mean() + beta.pow(2).mean()

    def forward(
        self,
        stiminds: torch.Tensor,  # (batch, trial_len, stimdim)
        embedding_rest: torch.Tensor | None = None,  # (embed_dim_rest,)
        embedding_stim: torch.Tensor | None = None,  # (embed_dim_stim,) or None
    ):
        batch_size = stiminds.shape[0]

        # flatten time × stimdim
        stiminds = stiminds.flatten(start_dim=1)  # (batch, trial_len*stimdim)

        # input layer + tanh
        hidden = self.in_layer(stiminds)
        hidden = self.in_activation(hidden)

        # FiLM after input layer
        if self.use_film:
            embedding_rest = embedding_rest.unsqueeze(0).repeat(batch_size, 1)
            embedding_stim = embedding_stim.unsqueeze(0).repeat(batch_size, 1)
            fused_embed = self.fuse_embeddings(
                embedding_rest, embedding_stim
            )  # (batch, mod_width)
            gamma_beta = self.film_head(fused_embed)  # (batch, 2*width)
            half = gamma_beta.size(-1) // 2
            gamma = gamma_beta[:, :half]
            beta = gamma_beta[:, half:]
            hidden = (1.0 + gamma) * hidden + beta

        # hidden stack (tanh as before)
        for layer in self.hiddens:
            hidden = self.hidden_activation(layer(hidden))

        # bases: (batch, time*num_bases)
        bases = self.out_layer(hidden)
        # bases: (batch, trial_len, num_bases)
        bases = bases.unflatten(dim=1, sizes=(self.trial_len, self.num_bases))
        return bases
