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

    def forward(self, data, embeddings_rest=None, embeddings_stim=None):
        """
        data: {session_id: (runway, covariates, y)}
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
        )

        forecast_decoded = self.ae.decode(latent_forecast)
        # y_hat = self.norms.inverse(forecast_decoded)
        y_hat = forecast_decoded
        return y_hat

    def normalize(self, y):
        return self.norms(y)
