"""
Some functions for testing/demo.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torch


def generate_ou_moving_mean(mean, trial_len=200, batch_size=1000, kappa=0.5, sigma=0.5):
    _, dt = np.linspace(0, trial_len, trial_len, retstep=True)

    X0 = np.random.normal(size=(batch_size,)) + mean[0].item()
    X = np.zeros((trial_len, batch_size))
    X[0, :] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(trial_len - 1, batch_size))

    # Uncomment for Euler Maruyama
    # for t in range(0,trial_len-1):
    #    X[t + 1, :] = X[t, :] + kappa*(mean[t] - X[t, :])*dt + sigma * np.sqrt(dt) * W[t, :]

    std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    for t in range(0, trial_len - 1):
        X[t + 1, :] = (
            mean[t].item()
            + np.exp(-kappa * dt) * (X[t, :] - mean[t].item())
            + std_dt * W[t, :]
        )

    X = torch.tensor(X)
    X = X.permute(1, 0).unsqueeze(-1)

    return X


def generate_ou_sinusoidal_moving_mean(
    trial_len=200,
    batch_size=1000,
    kappa=0.05,
    # Measured in time steps
    wavelength=40,
    # Shift the phase so all channels aren't the same mean
    # Measured in time steps
    phase_shift=0,
    sigma=0.5,
):
    lspace = torch.linspace(0, trial_len, trial_len)
    mean = torch.sin(lspace * 2 * math.pi / wavelength + phase_shift)

    # Cool; now OU on top of that:
    X = generate_ou_moving_mean(
        mean, trial_len=trial_len, batch_size=batch_size, kappa=kappa, sigma=sigma
    )

    return X


def bin_state_percentiles(y, yhat, ch=30, runway_length=20, bin_count=5):
    initial_states = y[:, runway_length, ch]
    recs = [(initial_states[idx], idx) for idx in range(initial_states.shape[0])]
    recs.sort()
    rec_count = len(recs)
    bin_size = int(rec_count / bin_count)

    idxs = []
    counts = []
    for ii in range(bin_count):
        rstart = ii * bin_size
        rend = rstart + bin_size

        cidxs = [r[1] for r in recs[rstart:rend]]

        idxs.append(cidxs)
        counts.append(len(cidxs))

    means = []
    meanshat = []
    minval = 1e99
    maxval = -1e99
    for cidxs in idxs:
        m = torch.mean(y[cidxs, :, ch], axis=0).detach().cpu().numpy()
        means.append(m)
        minval = min(minval, min(m))
        maxval = max(maxval, max(m))

        if yhat is not None:
            mhat = torch.mean(yhat[cidxs, :, ch], axis=0).detach().cpu().numpy()
            meanshat.append(mhat)
            minval = min(minval, min(m))
            maxval = max(maxval, max(m))

    return means, meanshat, minval, maxval


def graph_state_dependency(
    y,
    yhat,
    ch=0,
    runway_length=20,
    bin_count=5,
    title="Train",
    colormap=None,
    colormap_offset=0.0,
):
    means, meanshat, minval, maxval = bin_state_percentiles(
        y, yhat, ch=ch, runway_length=runway_length, bin_count=bin_count
    )

    if colormap:
        colors = getattr(plt.cm, colormap)(
            np.linspace(colormap_offset, 1, bin_count + 1)
        )
    else:
        colors = plt.cm.Paired(np.linspace(colormap_offset, 1, bin_count + 1))

    linewidth = 6.0
    plt.figure(figsize=(12, 8))
    for idx in range(bin_count):
        m = means[idx]
        color = colors[idx]
        plt.plot(
            m,
            color=color,
            linewidth=linewidth,
        )

        if yhat is not None:
            mhat = meanshat[idx]
            plt.plot(
                range(runway_length, mhat.shape[0] + runway_length),
                mhat,
                "--",
                color=color,
                linewidth=6.0,
            )

    plt.plot(
        [runway_length, runway_length],
        [1.1 * minval, 1.1 * maxval],
        "k--",
        linewidth=linewidth,
    )

    plt.title(title, fontsize=36)

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Time steps", fontsize=32)
    plt.tight_layout()
