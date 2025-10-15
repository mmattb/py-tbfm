"""
PyTorch Implementation of the basis generator network.
"""

import torch
from torch import nn


class Bases(nn.Module):
    """
    Basis generator network that operates on unpacked (non-temporal) inputs.

    Instead of taking (batch_size, trial_len, stimdim) inputs, takes (batch_size, stimdim).
    Still outputs (batch_size, trial_len, num_bases) like the regular Bases class.

    This is useful when you want to generate bases conditioned on stimulus features
    without using the temporal structure of the input.
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
            use_film: whether to use FiLM modulation
            embed_dim_rest: dimensionality of the rest data embedding, for FiLM
            embed_dim_stim: dimensionality of the stim data embedding, for FiLM
            device []: something tensor.to() would accept
        """
        super().__init__()

        self.num_bases = num_bases
        self.trial_len = trial_len
        self.latent_dim = latent_dim
        self.use_film = use_film
        self.stimdim = stimdim

        # Input is just stimdim (not flattened over time)
        in_dim = stimdim
        self.embed_dim_stim = embed_dim_stim
        self.embed_dim_rest = embed_dim_rest
        if use_film:
            self.embed_dim_film = embed_dim_rest + embed_dim_stim
            in_dim += self.embed_dim_film
        in_dim -= 1  # And we are throwing away the clock vec to replace it with FiLM
        self.in_dim = in_dim

        # Core MLP
        self.in_layer = nn.Linear(in_dim, latent_dim)
        self.in_activation = nn.Tanh()

        self.hiddens = nn.ModuleList()
        for _ in range(basis_depth):
            self.hiddens.append(nn.Linear(latent_dim, latent_dim))
        self.hidden_activation = nn.Tanh()

        # Output is trial_len * num_bases (we'll unflatten later)
        self.out_layer = nn.Linear(latent_dim, trial_len * num_bases)

        if use_film:
            # FiLM modulates the stimdim input (before MLP)
            self.proj_film = nn.Sequential(
                nn.Linear(self.embed_dim_film, 32),
                nn.Tanh(),
                nn.Linear(32, self.embed_dim_film),  # For now: this is just a transform
                nn.Tanh(),
            )
            # Initialize final layer to small weights (start near identity)
            nn.init.normal_(self.proj_film[2].weight, mean=0, std=0.01)
            nn.init.zeros_(self.proj_film[2].bias)
        else:
            self.proj_film = None

        self.to(device)

        self.last_hidden = None

        self._be_loud = False

    def be_loud(self):
        self._be_loud = True

    def get_reg_weights(self):
        total = 0.0
        for layer in [self.in_layer, *self.hiddens]:
            total = total + torch.linalg.norm(layer.weight)

        if self.use_film:
            total = total + torch.linalg.norm(self.proj_film[0].weight)
            total = total + torch.linalg.norm(self.proj_film[2].weight)

        return total

    def get_params(self):
        """Return core MLP params and FiLM params separately for optimization."""
        # core MLP inside Bases (exclude FiLM pieces)
        bases_core_params = [
            self.in_layer.parameters(),
            self.out_layer.parameters(),
        ]
        for layer in self.hiddens:
            bases_core_params.append(layer.parameters())

        bases_film_params = []
        if self.use_film:
            for layer in self.proj_film:
                bases_film_params.extend(list(layer.parameters()))

        return bases_core_params, bases_film_params

    def reset_state(self):
        """Reset cached state (for compatibility with Bases interface)."""
        self.last_hidden = None

    def forward(
        self,
        stiminds: torch.Tensor,  # (batch, stimdim) - NOTE: no trial_len dimension
        embedding_rest: torch.Tensor | None = None,  # (embed_dim_rest,)
        embedding_stim: torch.Tensor | None = None,  # (embed_dim_stim,) or None
    ):
        """
        Generate bases from unpacked stimulus descriptors.

        Args:
            stiminds: (batch_size, stimdim) - unpacked stimulus features
            embedding_rest: (embed_dim_rest,) - session-level rest embedding
            embedding_stim: (embed_dim_stim,) - learnable session/task embedding

        Returns:
            bases: (batch_size, trial_len, num_bases)
        """
        batch_size = stiminds.shape[0]

        if self.use_film:
            embedding_film = (
                torch.cat((embedding_rest, embedding_stim), dim=0)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            # film = self.proj_film(embedding_film)  # (batch, stimdim)

            if self._be_loud:
                print(
                    f"Embeddings shape: {embedding_film.shape}, "
                    # f"FiLM modulation - film shape: {film.shape}, "
                    f"stiminds shape: {stiminds.shape}"
                )
                self._be_loud = False

            # Apply FiLM modulation to stiminds
            # stiminds = torch.cat((stiminds, film), dim=-1)
            stiminds = torch.cat((stiminds, embedding_film), dim=-1)

        # Pass through MLP (stiminds is already (batch, stimdim))
        hidden = self.in_layer(stiminds)
        hidden = self.in_activation(hidden)

        # Hidden stack
        for layer in self.hiddens:
            hidden = self.hidden_activation(layer(hidden))

        self.last_hidden = hidden

        # Output: (batch, trial_len * num_bases)
        bases = self.out_layer(hidden)

        # Unflatten to (batch, trial_len, num_bases)
        bases = bases.unflatten(dim=1, sizes=(self.trial_len, self.num_bases))

        return bases
