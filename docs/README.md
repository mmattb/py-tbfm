<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Temporal Basis Function Model (TBFM), Supplementary Information
This page contains a variety of supplemental information which supports TBFM's publications

## Comparison models
We compare TBFMs to two existing model types. Details are as follows.

### Linear state space model (LSSM)
While there are a wide variety of LSSMs, some simple versions were previously demonstrated for modeling neural stimulation. These are commonly learned using the Kalman Filter, though they can also be trained using other methods such as backpropagation. In discrete time these simple LSSMs are specified as:

<img src="https://github.com/user-attachments/assets/debae1b5-2622-4410-9c3f-73365ddb5e55" height="60"/>

Here x is a latent state, u is the control input (e.g. stimulation parameters), and y is the prediction. Forward prediction can be performed by specifying an initial latent state x<sub>0</sub> and autoregressing forward in time.

We leverage this simple formulation by providing our stimulation descriptor as input. We estimate the initial state x<sub>0</sub> using the Moore-Penrose pseudoinverse of C and the last value of the runway. We train the model explicitly using backpropagation to perform multisstep forecasting. We use the L<sub>2</sub> prediction loss to train the three matrices of parameters A, B, C.

### LSTM-based dynamical systems model (ODE-LSTM)
We base this model on the more complex long short-term memory (LSTM) network, a nonlinear neural network for representing neural dynamics as a dynamical system with external inputs. The LSTM-based model uses an autoencoder architecture to lift the LFP data into a latent space, and
 predicts the effect of stimulation using an estimated dynamical system defined in the latent space.  The model predicts the change in neural activity between time steps, and performs forward prediction through a simple first-order integration. 

The latent space may be higher or lower dimensionality than the number of LFP channels.
In this demo case it is 96 dimensions - equal to the number of electrodes in the ECog array. We chose this dimensionality to ensure we were not
losing critical information when transforming into the latent space, but note that a lower dimensionality may provide similar results without penalty due to the inherently low dimensional nature of neural data. As in TBFM, we leverage a stimulation descriptor, which we concatenate to the estimated latent state of the system z.
The LSTM estimates a single step change in the system, which is summed
into the latent state to make a single step prediction z<sup>+</sup>. To make multi-step predictions the single step prediction is passed back into the dynamics model repeatedly. Thus: multi-step predictions are made using the first-order Euler integration method.

![ode_lstm](https://github.com/user-attachments/assets/72ae40a2-96db-4ba0-a390-e7403265544d)

We train the model on the same data sets as the TBFM. We crop random
sub-windows of size 60ms which we split into a 20ms runway and a
40ms prediction horizon. Like TBFM we leverage a multi-step MSE loss
function. Finally, we validate the model on the test set by performing
the full 164ms multistep prediction.

Training uses a tripartite loss function:

* an autoencoder loss  
  <img src="https://github.com/user-attachments/assets/65bf3d6e-973b-47ff-908f-64de815dcdbc" height="30"/>


* a dynamics prediction loss for all time steps in our window. Note however that we
    unroll predictions to make multi-step predictions, feeding our
    prediction back to the dynamics model at each step. The dynamics
    loss is a multi-step loss.  
  <img src="https://github.com/user-attachments/assets/001e07fe-e80f-4095-98d1-3e846093f7e9" height="30"/>

* a nearest-neighbor loss, which attempts to force all LFP values to align near each other in latent space so the model can take advantage of similar dynamics across channels and space. The simplest way to do that is to force the latent states to be centered at 0; hence:  
  <img src="https://github.com/user-attachments/assets/ab6a5a7d-721c-43db-b443-5ab2935f4e6f" height="30"/>

The LSTM model takes inspiration from dynamical systems and control methods which learn latent state representations for optimal control. Since it leverages an autoencoder architecture, it can be compared to methods such as deep Koopman for control, but without the key linearity constraint. We chose the LSTM model for comparison since it is a more expressive model than LSSMs, and has previously been demonstrated for control. While LSSMs were previously demonstrated for modeling neural stimulation, our ODE-LSTM model goes further by allowing for nonlinearities.

## Orthonormality penalty
We may optionally use an orthonormality penalty applied to the basis tensor. It is calculated as:

  <img src="https://github.com/user-attachments/assets/7d6f6113-33b7-4d01-a1c5-61a3218cebd4" height="500"/>

where B is the tensor of bases, and L_{ortho} is the calculated orthonormality penalty.

## Multisession model compilation
The following derives the compiled TBFM from a test-time adapted multi-session TBFM.
The compiled form is implemented in `TBFMMultisessionCompiled`
(`py-tbfm/tbfm/_multisession_module.py`).


<table>
<thead><tr><th>Symbol</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td>$C$</td><td>Number of channels in the session</td></tr>
<tr><td>$l$</td><td>AE latent dimension</td></tr>
<tr><td>$r$</td><td>Runway length (time steps)</td></tr>
<tr><td>$T$</td><td>Forecast horizon (time steps)</td></tr>
<tr><td>$b$</td><td>Number of basis vectors</td></tr>
<tr><td>$\mathbf{x} \in \mathbb{R}^{r \times C}$</td><td>Raw input runway, one trial</td></tr>
<tr><td>$\mathbf{Z} \in \mathbb{R}^{r \times l}$</td><td>Latent runway (after normalisation + encoding)</td></tr>
<tr><td>$B \in \mathbb{R}^{b \times T}$</td><td>Basis matrix (row = basis, col = time)</td></tr>
<tr><td>$W(\mathbf{Z}) \in \mathbb{R}^{l \times b}$</td><td>Basis weight matrix (latent channels &times; bases)</td></tr>
<tr><td>$W_{enc} \in \mathbb{R}^{l \times C}$</td><td>AE encoder weight</td></tr>
<tr><td>$b_{enc} \in \mathbb{R}^{l}$</td><td>AE encoder bias</td></tr>
<tr><td>$\boldsymbol{\alpha}, \boldsymbol{\beta} \in \mathbb{R}^C$</td><td>Per-channel normaliser scale and shift</td></tr>
<tr><td>$\tilde{W}_{enc} \in \mathbb{R}^{l \times C}$</td><td>Normaliser-folded encoder weight (IQR/Z-score absorbed)</td></tr>
<tr><td>$\tilde{b}_{enc} \in \mathbb{R}^{l}$</td><td>Normaliser-folded encoder bias</td></tr>
<tr><td>$c^{rest}_s \in \mathbb{R}^{3}$</td><td>Per-session resting-state context (A-ACF percentiles)</td></tr>
<tr><td>$c^{stim}_s \in \mathbb{R}^{15}$</td><td>Per-session stimulation context (optimised by TTA)</td></tr>
<tr><td>$\hat{\mathbf{y}} \in \mathbb{R}^{T \times C}$</td><td>Forecast in channel space</td></tr>
<tr><td>$Enc(\cdot)$, $Dec(\cdot)$</td><td>AE encoder and decoder</td></tr>
<tr><td>$\phi(\cdot)$</td><td>Activation: $\phi(P) = \text{rowNorm}(\tanh(P))$</td></tr>
</tbody>
</table>

---

### Full Forward Pass

For a fixed session $s$ after TTA, the full pipeline is:

$$
\mathbf{x}
\;\xrightarrow{\text{(1) normalise}}\;
\mathbf{x}_\text{norm}
\;\xrightarrow{\text{(2) } Enc}\;
\mathbf{Z}
\;\xrightarrow{\text{(3) basis-weight}}\;
W(\mathbf{Z})
\;\xrightarrow{\text{(4) contract with } B}\;
\hat{\mathbf{Z}}
\;\xrightarrow{\text{(5) } Dec}\;
\hat{\mathbf{y}}
$$

Steps (1)–(3) are affine in $\mathbf{x}$ and can be fused into a single
precomputed matrix. Step (4) uses a fixed $B$ (determined by the frozen basis
generator at $c^{rest}_s$, $c^{stim}_s$). Step (5) is a linear decode.
The only genuine nonlinearity is $\phi$ inside step (3).

---

### Step 1 — Normalisation

Both normaliser types (Z-score and quantile) are per-channel affine maps:

$$
\mathbf{x}_{\text{norm},t} = \mathbf{x}_t \odot \boldsymbol{\alpha} + \boldsymbol{\beta},
\qquad \boldsymbol{\alpha},\boldsymbol{\beta} \in \mathbb{R}^C
$$

**Z-score:** $\;\alpha_c = 1/\sigma_c,\quad \beta_c = -\mu_c/\sigma_c$

**Quantile:** $\;\alpha_c = 2/(q_{0.9,c} - q_{0.1,c}),\quad \beta_c = \alpha_c\cdot(-(q_{0.9,c}+q_{0.1,c})/2)$

---

### Step 2 — Linear Autoencoder Encoding

The AE encoder (`LinearChannelAE`) is a per-session affine map:

$$
\mathbf{z}_t = \mathbf{x}_{\text{norm},t}\, W_{enc}^\top + b_{enc}
$$

#### Folding the normaliser into the encoder

Substituting Step 1:

$$
\mathbf{z}_t
= (\mathbf{x}_t \odot \boldsymbol{\alpha} + \boldsymbol{\beta})\,W_{enc}^\top + b_{enc}
= \mathbf{x}_t\,\tilde{W}_{enc}^\top + \tilde{b}_{enc}
$$

where the **normalisation-folded encoder** (absorbing IQR or Z-score) is:

$$
\boxed{
\tilde{W}_{enc} = W_{enc} \odot \boldsymbol{\alpha}
\qquad
\tilde{b}_{enc} = \boldsymbol{\beta}\,W_{enc}^\top + b_{enc}
}
$$

($\boldsymbol{\alpha}$ is broadcast column-wise over $W_{enc}$, scaling column $c$ by $\alpha_c$.)

The full latent runway is then $\mathbf{Z} = \mathbf{x}\,\tilde{W}_{enc}^\top + \mathbf{1}_r \tilde{b}_{enc}^\top \in \mathbb{R}^{r \times l}$, a single affine map of the raw runway.

---

### Step 3 — Basis Weight Estimation

The `basis_weighting` layer is a linear map from the **flattened** latent runway
to a weight matrix:

$$
\text{vec}(W(\mathbf{Z})) = W_{bw}\,\text{vec}(\mathbf{Z}) + b_{bw},
\qquad W_{bw} \in \mathbb{R}^{lb \times rl},\quad b_{bw} \in \mathbb{R}^{lb}
$$

where $\text{vec}(\mathbf{Z}) \in \mathbb{R}^{rl}$ stacks all $r$ rows.

#### Fusing Steps 1–3

Substituting the latent encoding into the basis-weighting layer:

$$
\text{vec}(\mathbf{Z}) = (I_r \otimes \tilde{W}_{enc})\,\text{vec}(\mathbf{x}) + \mathbf{1}_r \otimes \tilde{b}_{enc}
$$

Hence,

$$
\text{vec}(W(\mathbf{Z}))
= \underbrace{W_{bw}(I_r \otimes \tilde{W}_{enc})}_{A_\text{pre}\;\in\;\mathbb{R}^{lb\times rC}}
  \text{vec}(\mathbf{x})
  \;+\;
  \underbrace{W_{bw}(\mathbf{1}_r \otimes \tilde{b}_{enc}) + b_{bw}}_{v_\text{pre}\;\in\;\mathbb{R}^{lb}}
$$

The Kronecker block structure is computed efficiently as:

$$
A_\text{pre}[:,\, tC:(t+1)C] = W_{bw}[:,\, tl:(t+1)l]\;\tilde{W}_{enc},
\quad t = 0,\ldots,r-1
$$

#### Nonlinearity

The raw weights are passed through $\phi$ (tanh + row-L2-normalise over the basis dimension):

$$
\phi\!\left(\text{reshape}(A_\text{pre}\,\text{vec}(\mathbf{x}) + v_\text{pre},\;(l,b))\right)
= \tilde{W}(\mathbf{x}) \in \mathbb{R}^{l \times b}
$$

$$
\phi(P)_{i,*} = \frac{\tanh(P_{i,*})}{\|\tanh(P_{i,*})\|_2},
\quad P \in \mathbb{R}^{l \times b}
$$

---

### Step 4 — Fixed Bases and $x_0$ Skip Connection

After TTA, $c^{stim}_s$ is fixed. The basis generator (conditioned on $c^{rest}_s$
and $c^{stim}_s$) produces a single constant matrix:

$$
B \in \mathbb{R}^{b \times T}
\quad \text{(fixed post-TTA)}
$$

The latent forecast is a weighted sum of basis vectors plus the $x_0$ skip:

$$
\hat{Z} = B^\top \tilde{W}(\mathbf{x})^\top + \mathbf{1}_T\,\mathbf{z}_{0}^\top
\;\in\; \mathbb{R}^{T \times l}
$$

where the $x_0$ skip encodes the last runway timestep through the
same normalisation-folded encoder:

$$
\mathbf{z}_0 = \mathbf{x}_{r}\,\tilde{W}_{enc}^\top + \tilde{b}_{enc} \;\in\; \mathbb{R}^l
$$

(This is row $r$ of $\mathbf{Z}$, so no extra computation is required.)

---

### Step 5 — AE Decoding

The `LinearChannelAE` uses tied weights: the decoder is the transpose of the encoder with no bias.

$$
\hat{\mathbf{y}} = \hat{Z}\,W_{enc}
$$

---

### Compiled Form (Summary)

<p>
After TTA with session $s$, stimulus condition with descriptor $s_s$, and
learnt contexts $c^{rest}_s$, $c^{stim}_s$, the full pipeline reduces to
<strong>five stored constant tensors</strong>
$\{A_\text{pre},\, v_\text{pre},\, B,\, \tilde{W}_{enc},\, \tilde{b}_{enc},\, W_{enc}\}$
and a <strong>single hidden layer</strong> with activation $\phi$:
</p>

$$
\boxed{
\hat{\mathbf{y}}
=
\Bigl(
  B^\top\,\phi\!\bigl(A_\text{pre}\,\text{vec}(\mathbf{x}) + v_\text{pre}\bigr)^\top
  +\,\mathbf{1}_T\mathbf{z}_0^\top
\Bigr)\,W_{enc}
}
$$

<p>with $\mathbf{z}_0 = \mathbf{x}_r\,\tilde{W}_{enc}^\top + \tilde{b}_{enc}$, where the
normalisation (IQR or Z-score) is absorbed into the folded encoder:</p>

$$
\tilde{W}_{enc} = W_{enc} \odot \boldsymbol{\alpha},
\qquad
\tilde{b}_{enc} = \boldsymbol{\beta}\,W_{enc}^\top + b_{enc}
$$

<p>The model is <strong>not affine</strong> (because $\phi$ contains $\tanh$), but it is a
single-hidden-layer network. All of: the normaliser, AE encoder,
<code>basis_weighting</code> layer, fixed bases, and AE decoder have been absorbed into
constant matrices. At inference only two matrix multiplies plus the $\phi$
activation are performed at runtime.</p>

#### Dimension summary

<table>
<thead><tr><th>Tensor</th><th>Shape</th><th>Formed from</th></tr></thead>
<tbody>
<tr><td>$A_\text{pre}$</td><td>$lb \times rC$</td><td>$W_{bw}$, $\tilde{W}_{enc}$ (Kronecker product)</td></tr>
<tr><td>$v_\text{pre}$</td><td>$lb$</td><td>$W_{bw}$, $\tilde{b}_{enc}$, $b_{bw}$</td></tr>
<tr><td>$B$</td><td>$b \times T$</td><td>Frozen basis generator at $c^{rest}_s$, $c^{stim}_s$, $s_s$</td></tr>
<tr><td>$\tilde{W}_{enc}$</td><td>$l \times C$</td><td>IQR/Z-score normaliser folded into $W_{enc}$</td></tr>
<tr><td>$\tilde{b}_{enc}$</td><td>$l$</td><td>IQR/Z-score normaliser folded into $b_{enc}$</td></tr>
<tr><td>$W_{enc}$</td><td>$l \times C$</td><td>AE encoder weight (= decoder weight, tied)</td></tr>
</tbody>
</table>

---

*See `TBFMMultisessionCompiled` and `TBFMMultisession.compile()` in
`py-tbfm/tbfm/_multisession_module.py` for the implementation.*
