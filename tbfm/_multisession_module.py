import torch
import torch.nn.functional as F
from torch import nn


class TBFMAdaptedCompiled(nn.Module):
    r"""
    Maximally-compiled inference-only form of TBFMMultisession for a single
    session and fixed stimulus condition (post-TTA).

    All purely affine stages — normaliser, AE encoder, ``basis_weighting``,
    fixed bases, AE decoder — are fused into precomputed constant tensors.
    The only remaining nonlinearity is ``tanh`` + row-L2-normalise on the
    basis weights.

    **Compiled form**:

    .. math::

        \hat{\mathbf{y}} =
        \Bigl(
          B^\top\,\phi(A_{\text{pre}}\,\text{vec}(\mathbf{x}) + v_{\text{pre}})^\top
          + \mathbf{1}_T \mathbf{z}_0^\top
        \Bigr)\,W_{enc}

    where :math:`\phi(P) = \text{rowNorm}(\tanh(P))` and
    :math:`\mathbf{z}_0 = \mathbf{x}_r\,\tilde{W}_{enc}^\top + \tilde{b}_{enc}`.

    Args:
        A_pre:       ``(l*b, r*C)`` fused preprocessing matrix.
        v_pre:       ``(l*b,)`` fused preprocessing bias.
        B:           ``(b, T)`` fixed basis matrix.
        W_enc_tilde: ``(l, C)`` normaliser-folded encoder weight.
        b_enc_tilde: ``(l,)`` normaliser-folded encoder bias.
        W_enc_hat:   ``(l, C)`` AE encoder weight (= decoder weight, tied).
        b_dec:       ``(C,)`` decoder bias.
        l:           AE latent dim / TBFM in_dim.
        b:           num_bases.
    """

    def __init__(
        self,
        A_pre: torch.Tensor,
        v_pre: torch.Tensor,
        B: torch.Tensor,
        W_enc_tilde: torch.Tensor,
        b_enc_tilde: torch.Tensor,
        W_enc_hat: torch.Tensor,
        b_dec: torch.Tensor,
        l: int,
        b: int,
    ):
        super().__init__()
        self.register_buffer("A_pre", A_pre)
        self.register_buffer("v_pre", v_pre)
        self.register_buffer("B", B)
        self.register_buffer("W_enc_tilde", W_enc_tilde)
        self.register_buffer("b_enc_tilde", b_enc_tilde)
        self.register_buffer("W_enc_hat", W_enc_hat)
        self.register_buffer("b_dec", b_dec)
        self.l = l
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw input runway ``(batch, r, C)``
        Returns:
            y_hat: forecast ``(batch, T, C)``
        """
        # Fused affine pre-block → basis-weight logits
        p = x.flatten(1) @ self.A_pre.T + self.v_pre        # (batch, l*b)
        W_tilde = p.unflatten(1, (self.l, self.b))           # (batch, l, b)

        # phi: tanh + row-L2-normalise
        W_tilde = torch.tanh(W_tilde)
        W_tilde = F.normalize(W_tilde, p=2, dim=-1)          # (batch, l, b)

        # Contract with fixed bases: (batch, l, b) @ (b, T) → (batch, T, l)
        Z_hat = (W_tilde @ self.B).permute(0, 2, 1)          # (batch, T, l)

        # z0 skip: encode last runway timestep
        z0 = x[:, -1, :] @ self.W_enc_tilde.T + self.b_enc_tilde  # (batch, l)
        Z_hat = Z_hat + z0.unsqueeze(1)                            # (batch, T, l)

        # AE decode: y_hat = Z_hat W_enc_hat + b_dec
        y_hat = Z_hat @ self.W_enc_hat + self.b_dec                # (batch, T, C)
        return y_hat


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

    def forward_reconstruct(self, data):
        runways = {sid: d[0] for sid, d in data.items()}
        runways_normalized = self.norms(runways)
        runways_latent = self.ae.encode(runways_normalized)
        runways_decoded = self.ae.decode(runways_latent)
        return runways_normalized, runways_decoded

    @torch.no_grad()
    def compile(
        self,
        session_id: str,
        stiminds: torch.Tensor,
        embedding_rest: torch.Tensor,
        embedding_stim: torch.Tensor | None = None,
    ) -> "TBFMAdaptedCompiled":
        """
        Compile the full model down to its minimal inference form for a single
        session and a fixed stimulus condition (typically called post-TTA).

        All purely affine stages — normaliser, AE encoder, basis_weighting,
        fixed bases, AE decoder — are fused into precomputed constant tensors.
        The only remaining nonlinearity is tanh + row-L2-normalise on the
        basis weights.

        Args:
            session_id: The session to compile for.
            stiminds: Stimulus descriptor ``(stimdim,)`` or ``(1, stimdim)``.
            embedding_rest: Post-TTA rest embedding ``(embed_dim_rest,)``.
            embedding_stim: Post-TTA stim embedding, or ``None``.

        Returns:
            TBFMAdaptedCompiled with no trainable parameters.
        """
        from .ae import SessionDispatcherLinearAE
        from .normalizers import ScalerZscore, ScalerQuant

        # 1. Normaliser → alpha/beta per-channel affine maps
        norm = self.norms.instances[session_id]
        if isinstance(norm, ScalerZscore):
            alpha = (1.0 / norm.std).squeeze(0)           # (C,)
            beta  = (-norm.mean / norm.std).squeeze(0)    # (C,)
        elif isinstance(norm, ScalerQuant):
            alpha = norm.a                                 # (C,)
            beta  = norm.a * norm.b                       # (C,)
        else:
            raise NotImplementedError(
                f"compile() does not support normalizer type {type(norm)}"
            )

        # 2. AE encoder weight and bias
        ae = self.ae
        if not isinstance(ae, SessionDispatcherLinearAE):
            raise NotImplementedError(
                f"compile() only supports SessionDispatcherLinearAE, got {type(ae)}"
            )
        inst  = ae.instances[session_id]
        W_enc = inst.w_enc   # (l, C)
        b_enc = inst.b_enc if inst.b_enc is not None else torch.zeros(
            W_enc.shape[0], device=W_enc.device, dtype=W_enc.dtype)
        b_dec_val = torch.zeros(
            W_enc.shape[1], device=W_enc.device, dtype=W_enc.dtype)

        # 3. Fold normaliser into encoder
        W_enc_tilde = W_enc * alpha           # (l, C)
        b_enc_tilde = beta @ W_enc.T + b_enc  # (l,)

        # 4. Fixed bases from frozen basis generator
        tbfm_inst = self.model.instances[session_id]
        if stiminds.dim() == 1:
            stiminds = stiminds.unsqueeze(0)  # (1, stimdim)
        tbfm_inst.eval()
        tbfm_inst.reset_state()
        B_batched = tbfm_inst.bases(
            stiminds,
            embedding_rest=embedding_rest,
            embedding_stim=embedding_stim,
        )   # (1, T, b)
        B = B_batched.squeeze(0).T   # (b, T)

        # 5. Fuse W_enc_tilde + basis_weighting into A_pre / v_pre
        W_bw  = tbfm_inst.basis_weighting.weight   # (l*b, r*l)
        b_bw  = tbfm_inst.basis_weighting.bias     # (l*b,)
        l_dim = tbfm_inst.in_dim
        b_num = tbfm_inst.num_bases
        r_len = W_bw.shape[1] // l_dim

        W_bw_3d  = W_bw.view(l_dim * b_num, r_len, l_dim)   # (l*b, r, l)
        A_pre_3d = W_bw_3d @ W_enc_tilde                     # (l*b, r, C)
        A_pre    = A_pre_3d.reshape(l_dim * b_num, r_len * W_enc_tilde.shape[1])  # (l*b, r*C)

        b_rep = b_enc_tilde.repeat(r_len)   # (r*l,)
        v_pre = W_bw @ b_rep + b_bw         # (l*b,)

        return TBFMAdaptedCompiled(
            A_pre=A_pre,
            v_pre=v_pre,
            B=B,
            W_enc_tilde=W_enc_tilde,
            b_enc_tilde=b_enc_tilde,
            W_enc_hat=W_enc,
            b_dec=b_dec_val,
            l=l_dim,
            b=b_num,
        )

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
