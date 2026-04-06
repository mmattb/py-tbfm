import torch
import torch.nn.functional as F
from torch import nn


class TBFMMultisession(nn.Module):
    def __init__(
        self,
        norms,
        aes,
        model,
        device=None,
    ):
        super().__init__()
        self.norms = norms
        self.ae = aes
        self.model = model
        self.device = device

    def forward(self, data, embeddings_rest=None, embeddings_stim=None, support_contexts=None):
        """
        data: {session_id: (runway, covariates, y)}
        embeddings_rest: {session_id: rest_embedding}
        embeddings_stim: {session_id: stim_embedding} (for MAML)
        support_contexts: {session_id: context_vector} (for hypernetwork)
        """
        # Unpack runways
        runways = {sid: d[0] for sid, d in data.items()}
        runways_normalized = self.norms(runways)
        runways_latent = self.ae.encode(runways_normalized)

        # Unpack covariates; go for gold.
        covariates = {sid: d[1] for sid, d in data.items()}
        latent_forecast = self.model(
            runways_latent,
            covariates,
            embeddings_rest=embeddings_rest,
            embeddings_stim=embeddings_stim,
            support_contexts=support_contexts,
        )

        forecast_decoded = self.ae.decode(latent_forecast)
        # y_hat = self.norms.inverse(forecast_decoded)
        y_hat = forecast_decoded
        return y_hat

    def forward_reconstruct(self, data):
        runways = {sid: d[0] for sid, d in data.items()}
        runways_normalized = self.norms(runways)
        runways_latent = self.ae.encode(runways_normalized)
        runways_decoded = self.ae.decode(runways_latent)
        return runways_normalized, runways_decoded

    def normalize(self, y):
        return self.norms(y)

    def eval(self, ae=True):
        """
        Set model to eval mode.
        
        Args:
            ae: If True, set AE to eval mode. If False, keep AE in training mode.
        """
        self.model.eval()
        self.norms.eval()
        if ae:
            self.ae.eval()
        else:
            self.ae.train()
        return self
    
    def train(self, mode=True, ae=True):
        """
        Set model to train mode.
        
        Args:
            mode: If True, set to train mode. If False, set to eval mode.
            ae: If True, apply mode to AE. If False, keep AE in opposite mode.
        """
        if mode:
            self.model.train()
            self.norms.train()
            if ae:
                self.ae.train()
            else:
                self.ae.eval()
        else:
            # If mode=False, call eval
            self.eval(ae=ae)
        return self


class TBFMMultisessionCompiled(nn.Module):
    """TBFMMultisession with prerendered bases for a specific session.

    Precomputes the bases matrix (output of the basis generator MLP + LoRA/residual)
    once at construction time, mirroring the TBFMCompiled pattern for vanilla TBFM.
    At inference, the basis generator and all LoRA/residual net forward passes are
    skipped; only norm → AE encode → basis_weighting matmul → AE decode remain.

    Args:
        ms_model: a TBFMMultisession instance (eval mode)
        session_id: the session for which to prerender bases
        stiminds_ref: (1, stimdim) tensor — reference stimulus descriptor used to
            prerender the bases (typically taken from the first trial)
        emb_rest: rest embedding for session_id
        emb_stim: stim embedding for session_id
    """

    def __init__(self, ms_model, session_id, stiminds_ref, emb_rest, emb_stim):
        super().__init__()
        self.device = ms_model.device
        self.norms = ms_model.norms
        self.ae = ms_model.ae
        self.session_id = session_id

        tbfm_instance = ms_model.model.instances[session_id]
        self.basis_weighting = tbfm_instance.basis_weighting
        self.in_dim = tbfm_instance.in_dim
        self.num_bases = tbfm_instance.num_bases

        # Prerender bases: (1, trial_len, num_bases)
        with torch.no_grad():
            bases = tbfm_instance.bases(
                stiminds_ref,
                embedding_rest=emb_rest,
                embedding_stim=emb_stim,
            )
            self.register_buffer("prerendered_bases", bases.detach())

    def forward(self, data, **ignored):
        sid = self.session_id
        runway_raw = data[sid][0]

        runways_normalized = self.norms({sid: runway_raw})
        runways_latent = self.ae.encode(runways_normalized)
        runway_latent = runways_latent[sid]

        x0 = runway_latent[:, -1:, :]
        batch_size = runway_latent.shape[0]

        basis_weights = self.basis_weighting(runway_latent.flatten(start_dim=1))
        basis_weights = basis_weights.unflatten(1, (self.in_dim, self.num_bases))
        basis_weights = torch.tanh(basis_weights)
        basis_weights = F.normalize(basis_weights, p=2, dim=-1)

        # Expand prerendered bases to batch size
        bases = self.prerendered_bases.expand(batch_size, -1, -1)
        latent_preds = (basis_weights @ bases.permute(0, 2, 1)).permute(0, 2, 1) + x0

        forecast = self.ae.decode({sid: latent_preds})
        return forecast
