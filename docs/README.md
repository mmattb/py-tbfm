# Temporal Basis Function Model (TBFM), Supplementary Information
This page contains a variety of supplemental information which supports TBFM's publications

## Comparison models
We compare TBFMs to two existing model types. Details are as follows.

### Linear state space model (LSSM)
While there are a wide variety of LSSMs, some simple versions were previously demonstrated for modeling neural stimulation. These are commonly learned using the Kalman Filter, though they can also be trained using other methods such as backpropagation. In discrete time these simple LSSMs are specified as:

$x_{k+1} = Ax_k + Bu_k$

$y_k = Cx_k$

Here $x$ is a latent state, $u$ is the control input (e.g. stimulation parameters), and $y$ is the prediction. Forward prediction can be performed by specifying an initial latent state $x_0$ and autoregressing forward in time.

We leverage this simple formulation by providing our stimulation descriptor as input. We estimate the initial state $x_0$ using the Moore-Penrose pseudoinverse of $C$ and the last value of the runway. We train the model explicitly using backpropagation to perform multisstep forecasting. We use the $L_2$ prediction loss to train the three matrices of parameters $A, B, C$.

### LSTM-based dynamical systems model (ODE-LSTM}
We base this model on the more complex long short-term memory (LSTM) network, a nonlinear neural network for representing neural dynamics as a dynamical system with external inputs. The LSTM-based model uses an autoencoder architecture to lift the LFP data into a latent space, and
 predicts the effect of stimulation using an estimated dynamical system defined in the latent space.  The model predicts the change in neural activity between time steps, and performs forward prediction through a simple first-order integration. 

The latent space may be higher or lower dimensionality than the number of LFP channels.
In this demo case it is 96 dimensions - equal to the number of electrodes in the ECog array. We chose this dimensionality to ensure we were not
losing critical information when transforming into the latent space, but note that a lower dimensionality may provide similar results without penalty due to the inherently low dimensional nature of neural data. As in TBFM, we leverage a stimulation descriptor, which we concatenate to the estimated latent state of the system $z$.
The LSTM estimates a single step change in the system, which is summed
into the latent state to make a single step prediction $z^+$. To make multi-step predictions the single step prediction is passed back into the dynamics model repeatedly. Thus: multi-step predictions are made using the first-order Euler integration method.

![ode_lstm](https://github.com/user-attachments/assets/72ae40a2-96db-4ba0-a390-e7403265544d)

We train the model on the same data sets as the TBFM. We crop random
sub-windows of size 60ms which we split into a 20ms runway and a
40ms prediction horizon. Like TBFM we leverage a multi-step MSE loss
function. Finally, we validate the model on the test set by performing
the full 164ms multistep prediction.

Training uses a tripartite loss function:

* an autoencoder loss $L_2(x, g^{-1}(g(x)))$
* a dynamics prediction loss $L_2(z_t + \Delta z, z_{t+1})$ for all time steps in our window. Note however that we
    unroll predictions to make multi-step predictions, feeding our
    prediction back to the dynamics model at each step. The dynamics
    loss is a multi-step loss.
* a nearest-neighbor loss, which attempts to force all LFP values to align near each other in latent space so the model can take advantage of similar dynamics across channels and space. The simplest way to do that is to force the latent states to be centered at 0; hence $L_2(\overline{z}, 0)$.

The LSTM model takes inspiration from dynamical systems and control methods which learn latent state representations for optimal control. Since it leverages an autoencoder architecture, it can be compared to methods such as deep Koopman for control, but without the key linearity constraint. We chose the LSTM model for comparison since it is a more expressive model than LSSMs, and has previously been demonstrated for control. While LSSMs were previously demonstrated for modeling neural stimulation, our ODE-LSTM model goes further by allowing for nonlinearities.
