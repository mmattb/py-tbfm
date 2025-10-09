"""
PyTorch Implementation of the basis generator network.
"""

import torch
from torch import nn

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
        # self.hidden_activation = nn.GELU()
        self.hidden_activation = nn.Tanh()

        self.out_layer = nn.Linear(latent_dim, trial_len * num_bases)

        self.embed_dim_stim = embed_dim_stim
        self.embed_dim_rest = embed_dim_rest
        if use_film:
            self.embed_dim_film = embed_dim_rest + embed_dim_stim
            self.proj_film = nn.Sequential(
                nn.Linear(self.embed_dim_film, 32),
                nn.Tanh(),
                nn.Linear(32, trial_len),
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
        # core MLP inside Bases (exclude FiLM pieces)
        bases_core_params = []
        bases_core_params += [
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
        self.last_hidden = None

    def forward(
        self,
        stiminds: torch.Tensor,  # (batch, trial_len, stimdim)
        embedding_rest: torch.Tensor | None = None,  # (embed_dim_rest,)
        embedding_stim: torch.Tensor | None = None,  # (embed_dim_stim,) or None
    ):
        batch_size = stiminds.shape[0]

        if self.use_film:
            embedding_film = (
                torch.cat((embedding_rest, embedding_stim), dim=0)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            film = self.proj_film(embedding_film)
            if self._be_loud:
                print(
                    "eeee",
                    film[0],
                    film.shape,
                    film[0].shape,
                    stiminds.shape,
                )
                self._be_loud = False

            stiminds = stiminds.clone()
            stiminds[:, :, -1] = stiminds[:, :, -1] + film

        # flatten time × stimdim
        stiminds = stiminds.flatten(start_dim=1)  # (batch, trial_len*stimdim)

        # input layer + tanh
        hidden = self.in_layer(stiminds)
        hidden = self.in_activation(hidden)

        # hidden stack (tanh as before)
        for layer in self.hiddens:
            hidden = self.hidden_activation(layer(hidden))

        self.last_hidden = hidden

        # bases: (batch, time*num_bases)
        bases = self.out_layer(hidden)
        # bases: (batch, trial_len, num_bases)
        bases = bases.unflatten(dim=1, sizes=(self.trial_len, self.num_bases))
        return bases
