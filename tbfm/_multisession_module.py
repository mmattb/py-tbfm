import torch
import torch.nn.functional as F
from torch import nn


class TBFMMultisessionCompiled(nn.Module):
    r"""
    Maximally-compiled inference-only form of TBFMMultisession for a single session
    and fixed stimulus condition (post-TTA).

    Mathematical Derivation
    -----------------------

    **Notation**

    * :math:`\mathbf{x} \in \mathbb{R}^{r \times C}` — raw input runway
    * :math:`\boldsymbol{\alpha}, \boldsymbol{\beta} \in \mathbb{R}^C` — per-channel normaliser scale/shift
      (:math:`x_{\text{norm},t} = x_t \odot \alpha + \beta`; Z-score or IQR)
    * :math:`W_{enc} \in \mathbb{R}^{l \times C}` — AE encoder weight (``LinearChannelAE``, tied decoder)
    * :math:`b_{enc} \in \mathbb{R}^l` — AE encoder bias
    * :math:`\tilde{W}_{enc} = W_{enc} \odot \boldsymbol{\alpha}` — normaliser-folded encoder weight
    * :math:`\tilde{b}_{enc} = \boldsymbol{\beta}\,W_{enc}^\top + b_{enc}` — normaliser-folded encoder bias
    * :math:`W_{bw} \in \mathbb{R}^{lb \times rl}`,
      :math:`b_{bw} \in \mathbb{R}^{lb}` — TBFM ``basis_weighting`` linear layer
    * :math:`B \in \mathbb{R}^{b \times T}` — fixed bases (precomputed from frozen
      basis generator at the post-TTA embeddings and fixed stim descriptor)
    * :math:`b` = num\_bases, :math:`l` = AE latent / TBFM in\_dim

    **Step 1 — Normalise + AE-encode per timestep (affine, same map for every** :math:`t`\ **)**

    .. math::

        z_t = (x_t \odot \alpha + \beta)\, W_{enc}^\top + b_{enc}
            = x_t\, \tilde{W}_{enc}^\top + \tilde{b}_{enc}

    where the *normaliser-folded encoder* (IQR or Z-score absorbed) is:

    .. math::

        \tilde{W}_{enc} = W_{enc} \odot \alpha \quad (\text{column } c
            \text{ of } W_{enc} \text{ scaled by } \alpha_c)

        \tilde{b}_{enc} = \beta\, W_{enc}^\top + b_{enc} .

    **Step 2 — basis\_weighting (affine in the flattened latent runway)**

    .. math::

        \mathbf{p} = W_{bw}\,\text{vec}(\mathbf{h}) + b_{bw}
                   \in \mathbb{R}^{d_z K}

    where :math:`\text{vec}(\mathbf{h})` stacks all :math:`T_r` rows of
    :math:`\mathbf{h}` into a vector of length :math:`T_r d_z`.

    **Fusing Steps 1–2 into a single matrix–vector product**

    The per-timestep encoding writes
    :math:`\text{vec}(\mathbf{h}) = (I_{T_r} \otimes W_{\text{eff}})\,
    \text{vec}(\mathbf{x}) + \mathbf{1}_{T_r} \otimes b_{\text{eff}}`.
    Substituting into Step 2:

    .. math::

        \mathbf{p} = \underbrace{W_{bw}(I_{T_r} \otimes W_{\text{eff}})}_{
            M_{\text{pre}} \;\in\; \mathbb{R}^{d_z K \times T_r C}}
            \text{vec}(\mathbf{x})
            + \underbrace{W_{bw}(\mathbf{1}_{T_r} \otimes b_{\text{eff}}) + b_{bw}}_{
            v_{\text{pre}} \;\in\; \mathbb{R}^{d_z K}}

    In practice the Kronecker block-matrix multiply is computed as

    .. math::

        M_{\text{pre}}[:, t C : (t+1)C] = W_{bw}[:, t d_z : (t+1)d_z]\, W_{\text{eff}}

    which is a single call to ``torch.einsum`` or ``view``/``matmul``.

    **Step 3 — Only nonlinearity**

    .. math::

        \tilde{W} = \phi(\text{reshape}(\mathbf{p},\;(l,b)))
        \quad \text{where} \quad
        \phi(P) = \text{rowNorm}(\tanh(P)) \in \mathbb{R}^{l \times b}

    (``tanh`` elementwise; ``rowNorm`` = L2-normalise each row over the :math:`b`
    basis dimension, i.e. ``F.normalize(..., p=2, dim=-1)``).

    **Step 4 — Contract with fixed bases, add** :math:`z_0` **skip, decode**

    .. math::

        \hat{\mathbf{y}} =
          \Bigl( B^\top\,\tilde{W}^\top + \mathbf{1}_T \mathbf{z}_0^\top \Bigr)\,W_{enc}

    where the :math:`z_0` skip encodes the last runway timestep using the
    *normalisation-folded* encoder (no basis-weighting):

    .. math::

        \mathbf{z}_0 = \mathbf{x}_r\, \tilde{W}_{enc}^\top + \tilde{b}_{enc}
            \in \mathbb{R}^l

    **Summary — compiled form (five stored constants + one nonlinearity)**

    .. math::

        \boxed{
            \hat{\mathbf{y}} =
            \Bigl(
              B^\top\,\phi(A_{\text{pre}}\,\text{vec}(\mathbf{x}) + v_{\text{pre}})^\top
              + \mathbf{1}_T \mathbf{z}_0^\top
            \Bigr)\,W_{enc}
        }

    The model is provably *not* affine because :math:`\phi` contains ``tanh``.
    However it is a **single-hidden-layer network** with a tanh+row-normalise
    activation; all other stages (normaliser, AE encode, basis_weighting, fixed
    bases, AE decode) collapse into constant matrices.

    Args:
        A_pre:       ``(l*b, r*C)`` fused preprocessing matrix.
        v_pre:       ``(l*b,)`` fused preprocessing bias.
        B:           ``(b, T)`` fixed basis matrix (post-TTA, post-basis-generator).
        W_enc_tilde: ``(l, C)`` normaliser-folded encoder weight.
        b_enc_tilde: ``(l,)`` normaliser-folded encoder bias.
        W_enc_hat:   ``(l, C)`` AE encoder weight (= decoder weight, tied).
        b_dec:       ``(C,)`` decoder bias (zeros for ``LinearChannelAE``).
        l:           int, AE latent dim / TBFM in_dim.
        b:           int, num_bases.
        normalize_weights: if ``True`` apply tanh + row-L2-normalise in :math:`\phi`.
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
        normalize_weights: bool = True,
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
        self.normalize_weights = normalize_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw input runway ``(batch, r, C)``

        Returns:
            y_hat: forecast ``(batch, T, C)``
        """
        # Fused linear pre-block → basis-weight logits: A_pre vec(x) + v_pre
        p = x.flatten(1) @ self.A_pre.T + self.v_pre        # (batch, l*b)
        W_tilde = p.unflatten(1, (self.l, self.b))           # (batch, l, b)

        # phi: tanh + row-L2-normalise (W(Z) in manuscript)
        if self.normalize_weights:
            W_tilde = torch.tanh(W_tilde)
            W_tilde = F.normalize(W_tilde, p=2, dim=-1)      # (batch, l, b)

        # Contract with fixed bases B: (batch, l, b) @ (b, T) → (batch, T, l)
        Z_hat = (W_tilde @ self.B).permute(0, 2, 1)          # (batch, T, l)

        # z0 skip: encode last runway timestep (z0 = x_r W_enc_tilde^T + b_enc_tilde)
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

    @torch.no_grad()
    def compile(
        self,
        session_id: str,
        stiminds: torch.Tensor,
        embedding_rest: torch.Tensor,
        embedding_stim: torch.Tensor | None = None,
    ) -> "TBFMMultisessionCompiled":
        r"""
        Compile the full model down to its minimal inference form for a single
        session and a *fixed* stimulus condition (typically called post-TTA).

        All purely affine stages — normaliser, AE encoder, ``basis_weighting``,
        fixed bases, AE decoder — are fused into precomputed constant tensors.
        The only remaining nonlinearity is the ``tanh`` + row-L2-normalise on the
        basis weights.  See ``TBFMMultisessionCompiled`` for the full derivation.

        Args:
            session_id: The session to compile for.
            stiminds: Stimulus descriptor for the fixed condition.
                ``(stimdim,)`` or ``(1, stimdim)`` — passed directly to the
                basis generator.
            embedding_rest: Post-TTA rest embedding ``(embed_dim_rest,)``.
            embedding_stim: Post-TTA stim embedding ``(embed_dim_stim,)`` or
                ``None`` if the model does not use stim embeddings.

        Returns:
            ``TBFMMultisessionCompiled`` — a lightweight ``nn.Module`` with no
            trainable parameters and a single ``forward(x)`` method.
        """
        # Lazy imports to avoid potential circular-import issues at module load
        from .ae import SessionDispatcherLinearAE
        from .normalizers import ScalerZscore, ScalerQuant

        # ------------------------------------------------------------------ #
        # 1. Normaliser: alpha/beta per-channel affine maps                  #
        #    Z-score:  alpha = 1/sigma,  beta = -mu/sigma                    #
        #    IQR:      alpha = 2/(q90-q10), beta = -alpha*(q90+q10)/2        #
        # ------------------------------------------------------------------ #
        norm = self.norms.instances[session_id]

        if isinstance(norm, ScalerZscore):
            alpha = (1.0 / norm.std).squeeze(0)          # (C,)
            beta  = (-norm.mean / norm.std).squeeze(0)   # (C,)
        elif isinstance(norm, ScalerQuant):
            alpha = norm.a                                # (C,)
            beta  = norm.a * norm.b                      # (C,)
        else:
            raise NotImplementedError(
                f"compile() does not support normalizer type {type(norm)}"
            )

        # ------------------------------------------------------------------ #
        # 2. AE encoder weight and bias (LinearChannelAE, tied decoder)      #
        # ------------------------------------------------------------------ #
        ae = self.ae

        if not isinstance(ae, SessionDispatcherLinearAE):
            raise NotImplementedError(
                f"compile() only supports SessionDispatcherLinearAE, got {type(ae)}"
            )

        inst  = ae.instances[session_id]
        W_enc = inst.w_enc   # (l, C)
        b_enc = inst.b_enc if inst.b_enc is not None else torch.zeros(
            W_enc.shape[0], device=W_enc.device, dtype=W_enc.dtype)   # (l,)
        b_dec_val = torch.zeros(
            W_enc.shape[1], device=W_enc.device, dtype=W_enc.dtype)   # (C,)

        # ------------------------------------------------------------------ #
        # 3. Fold normaliser into encoder: W_enc_tilde, b_enc_tilde          #
        #    z_t = (x_t * alpha + beta) @ W_enc.T + b_enc                   #
        #        = x_t @ W_enc_tilde.T + b_enc_tilde                        #
        #    W_enc_tilde = W_enc * alpha  (column-wise broadcast)            #
        #    b_enc_tilde = beta @ W_enc.T + b_enc                           #
        # ------------------------------------------------------------------ #
        W_enc_tilde = W_enc * alpha                  # (l, C)
        b_enc_tilde = beta @ W_enc.T + b_enc         # (l,)

        # ------------------------------------------------------------------ #
        # 4. Fixed bases from the frozen basis generator                     #
        # ------------------------------------------------------------------ #
        tbfm_inst = self.model.instances[session_id]

        if stiminds.dim() == 1:
            stiminds = stiminds.unsqueeze(0)   # (1, stimdim)

        tbfm_inst.eval()
        tbfm_inst.reset_state()
        B_batched = tbfm_inst.bases(
            stiminds,
            embedding_rest=embedding_rest,
            embedding_stim=embedding_stim,
        )   # (1, T_y, K)
        B = B_batched.squeeze(0).T   # (b, T) — row = basis, col = time

        # ------------------------------------------------------------------ #
        # 5. Fuse W_enc_tilde + basis_weighting into A_pre / v_pre           #
        #                                                                     #
        #    A_pre[:,tC:(t+1)C] = W_bw[:,tl:(t+1)l] @ W_enc_tilde           #
        #    v_pre = W_bw @ (1_r ⊗ b_enc_tilde) + b_bw                      #
        # ------------------------------------------------------------------ #
        W_bw  = tbfm_inst.basis_weighting.weight   # (l*b, r*l)
        b_bw  = tbfm_inst.basis_weighting.bias     # (l*b,)
        l_dim = tbfm_inst.in_dim
        b_num = tbfm_inst.num_bases
        r_len = W_bw.shape[1] // l_dim

        # A_pre[:, t*C:(t+1)*C] = W_bw[:, t*l:(t+1)*l] @ W_enc_tilde
        W_bw_3d  = W_bw.view(l_dim * b_num, r_len, l_dim)       # (l*b, r, l)
        A_pre_3d = W_bw_3d @ W_enc_tilde                         # (l*b, r, C)
        A_pre    = A_pre_3d.reshape(l_dim * b_num, r_len * W_enc_tilde.shape[1])  # (l*b, r*C)

        # v_pre = W_bw @ repeat(b_enc_tilde, r) + b_bw
        b_rep = b_enc_tilde.repeat(r_len)    # (r*l,)
        v_pre = W_bw @ b_rep + b_bw          # (l*b,)

        normalize_weights = tbfm_inst.normalize_weights

        return TBFMMultisessionCompiled(
            A_pre=A_pre,
            v_pre=v_pre,
            B=B,
            W_enc_tilde=W_enc_tilde,
            b_enc_tilde=b_enc_tilde,
            W_enc_hat=W_enc,
            b_dec=b_dec_val,
            l=l_dim,
            b=b_num,
            normalize_weights=normalize_weights,
        )
