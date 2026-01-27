import math
import os
import random

import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats
import torch
from torch import nn
from torcheval.metrics.functional import r2_score

import preprocess as pp
import data.dataset as ds
import utils


def get_bin_bounds(
    idxs, data, ch, runway, boundary_shrinkage_low=None, boundary_shrinkage_high=None
):
    # Get the bin boundaries for our N bins
    # [(lower, upper), ...]
    bin_bounds = []
    upper = None
    for bidx, i in enumerate(idxs):
        binsub = data[i, runway, ch]

        if upper is None:
            lower = min(binsub)
            # XXX assumes lower bound is negative; i.e. this shrunks towards 0
            if boundary_shrinkage_low:
                lower *= boundary_shrinkage_low
        else:
            lower = upper
        upper = max(binsub)
        if boundary_shrinkage_high and bidx == (len(idxs) - 1):
            upper *= boundary_shrinkage_high

        bounds = (lower.item(), upper.item())

        bin_bounds.append(bounds)
    return bin_bounds


def load_stim_and_rest(session_id, session_subdir, session_subdir_rest):
    dsetstim = ds.load_data_some_sessions(
        [
            session_id,
        ],
        session_subdir=session_subdir,
        batch_size=20000,
        window_size=184,
        in_memory=True,
        device="cpu",
    )
    dstim = next(iter(dsetstim))[1][0]
    dstim = utils.zscore_raw(dstim)
    timelen = dstim.shape[1]

    data_dir = os.path.join("data", session_id)
    drest = torch.load(
        os.path.join(data_dir, session_subdir_rest, "base.torch"), weights_only=False
    )
    drest = utils.zscore_raw(drest[:, :timelen, :])

    return dstim, drest


def get_stim_v_rest(
    session_id,
    ch,
    session_subdir="torchlowbp",
    session_subdir_rest="torchbaselow",
    boundary_shrinkage_low=None,
    boundary_shrinkage_high=None,
    runway=20,
    num_bins=5,
    get_micro=False,
    get_bounds=False,
    statistic="mean",  # {'mean', 'median'}
):
    dstim, drest = load_stim_and_rest(session_id, session_subdir, session_subdir_rest)

    num_ch = drest.shape[2]
    assert dstim.shape[2] == num_ch

    names, means, meanshat, minval, maxval, idxs, counts = (
        utils.split_highs_lows_percentiles(
            None, drest, ch=ch, runway=runway, bins=num_bins, statistic=statistic
        )
    )

    diff = torch.abs(dstim[:, runway, ch].unsqueeze(1) - drest[:, runway, ch])
    nearest_indices = torch.argmin(diff, dim=0)

    stim_bins = []
    for bidx, rest_idxs in enumerate(idxs):
        stim_idxs = nearest_indices[rest_idxs]
        stim_bins.append(dstim[stim_idxs, :, ch])

    means = [torch.tensor(m) for m in means]
    if statistic == "mean":
        means_stim = [torch.mean(sb, axis=0) for sb in stim_bins]
    elif statistic == "median":
        means_stim = [torch.median(sb, axis=0).values for sb in stim_bins]
    else:
        raise ValueError(f"Statistic must be one of 'mean', 'median'; got {statistic}")

    if get_micro:
        rest_micro = [drest[i, :, :] for i in idxs]
        return means, means_stim, rest_micro, stim_bins
    elif get_bounds:
        return means, means_stim, bin_bounds
    else:
        return means, means_stim


def graph_stim_v_rest(
    session_id,
    ch,
    session_subdir="torchlowbp",
    session_subdir_rest="torchbaselow",
    num_bins=5,
    runway=20,
    statistic="mean",  # {'mean', 'median'}
    boundary_shrinkage_low=None,
    boundary_shrinkage_high=None,
    show_diff=False,
    show_legend=True,
    colormap=None,
    axs=None,
    savepath=None,
):

    means, means_stim = get_stim_v_rest(
        session_id,
        ch,
        session_subdir=session_subdir,
        session_subdir_rest=session_subdir_rest,
        num_bins=num_bins,
        boundary_shrinkage_low=boundary_shrinkage_low,
        boundary_shrinkage_high=boundary_shrinkage_high,
        runway=runway,
        statistic=statistic,
    )

    if axs is None:
        axso = plt
        plt.figure(figsize=(12, 10))
    else:
        axso = axs

    cgrid = [r / num_bins for r in range(0, num_bins)]
    if colormap:
        colors = getattr(plt.cm, colormap)(cgrid)
    else:
        colors = plt.cm.cividis(cgrid)

    maxval = -1e99
    minval = 1e99
    for bidx, m in enumerate(means):
        m = means[bidx]
        mstim = means_stim[bidx]

        if show_diff:
            mdiff = mstim - m
            axso.plot(mdiff, color=colors[bidx], linewidth=6.0)
        else:
            axso.plot(m, "--", color=colors[bidx], linewidth=6.0)
            axso.plot(mstim, color=colors[bidx], linewidth=6.0)

        maxval = max(maxval, max(m))
        maxval = max(maxval, max(mstim))
        minval = min(minval, min(m))
        minval = min(minval, min(mstim))
    axso.plot(
        [runway, runway],
        [1.1 * minval, 1.1 * maxval],
        "k--",
        linewidth=6.0,
    )

    legend_color = "0.6"
    try:
        xmin, xmax = axso.xlim()
        ymin, ymax = axso.ylim()
    except AttributeError:
        xmin, xmax = axso.get_xlim()
        ymin, ymax = axso.get_ylim()

    if show_diff:
        label = "Stim response\n(Stim - Rest)"
        axso.plot(
            [500, 500], [500, 500], linewidth=6.0, color=legend_color, label=label
        )
    else:
        axso.plot(
            [500, 500],
            [500, 500],
            "--",
            linewidth=6.0,
            color=legend_color,
            label="Rest",
        )
        axso.plot(
            [500, 500], [500, 500], linewidth=6.0, color=legend_color, label="Stim"
        )

    try:
        axso.xlim([xmin, xmax])
        axso.ylim([ymin, ymax])
    except AttributeError:
        axso.set_xlim([xmin, xmax])
        axso.set_ylim([ymin, ymax])

    if show_legend:
        plt.legend(fontsize=34, loc="lower right")
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlabel("Time (ms)", fontsize=32)
        plt.ylabel("LFP (Z-scored)", fontsize=32)

    if axs is None:
        plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)

    if axs is None:
        plt.show()


def graph_stim_v_rest_session(
    session_id,
    session_subdir="torchlowbp",
    session_subdir_rest="torchbaselow",
    num_bins=5,
    runway=20,
    show=True,
    savepath=None,
):

    data_dir = os.path.join("data", session_id)
    drest = torch.load(
        os.path.join(data_dir, session_subdir_rest, "base.torch"), weights_only=True
    )
    num_ch = drest.shape[2]

    fig = plt.figure(figsize=(12, 2.5 * num_ch))

    for ch in range(num_ch):
        ax = fig.add_subplot(num_ch, 2, 1 + ch * 2)
        if ch == 0:
            show_legend = True
        else:
            show_legend = False
        ax.set_ylabel(f"ch {ch}")
        graph_stim_v_rest(
            session_id,
            ch,
            session_subdir=session_subdir,
            session_subdir_rest=session_subdir_rest,
            num_bins=num_bins,
            runway=runway,
            show_legend=show_legend,
            axs=ax,
        )
        ax = fig.add_subplot(num_ch, 2, ch * 2 + 2)
        graph_stim_v_rest(
            session_id,
            ch,
            session_subdir=session_subdir,
            session_subdir_rest=session_subdir_rest,
            num_bins=num_bins,
            runway=runway,
            show_diff=True,
            show_legend=show_legend,
            axs=ax,
        )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()


def graph_highs_lows_shared_bins(
    yhat,
    y,
    yhat_test,
    ytest,
    ch=30,
    runway=20,
    boundary_shrinkage_low=None,
    boundary_shrinkage_high=None,
    bins=5,
    savepath=None,
    colormap=None,
    colormap_offset=0.0,
    ylabel="LFP (Z-scored)",
):
    names, means, meanshat, minval, maxval, idxs, _ = (
        utils.split_highs_lows_percentiles(yhat, y, ch=ch, runway=runway, bins=bins)
    )

    bin_bounds = get_bin_bounds(
        idxs,
        y,
        ch,
        runway,
    )

    initial_test_states = ytest[:, runway, ch]
    test_bins = []
    testhat_bins = []
    for bidx, bounds in enumerate(bin_bounds):
        low, high = bounds
        if boundary_shrinkage_low and bidx == 0:
            print(low, low * boundary_shrinkage_low)
            low *= boundary_shrinkage_low
        if boundary_shrinkage_high and bidx == (len(bin_bounds) - 1):
            print(high, high * boundary_shrinkage_high)
            high *= boundary_shrinkage_high
        cidxs = torch.logical_and(
            initial_test_states >= low, initial_test_states <= high
        )
        test_bins.append(ytest[cidxs, :, ch])
        testhat_bins.append(yhat_test[cidxs, :, ch])

    means = [torch.tensor(m) for m in means]
    meanshat = [torch.tensor(m) for m in meanshat]
    means_test = [torch.mean(tb, axis=0) for tb in test_bins]
    means_testhat = [torch.mean(tb, axis=0) for tb in testhat_bins]

    minval = min(minval, min([torch.min(m).item() for m in means_test]))
    maxval = max(maxval, max([torch.max(m).item() for m in means_test]))

    bins = len(names)
    if colormap:
        colors = getattr(plt.cm, colormap)(np.linspace(colormap_offset, 1, bins + 1))
    else:
        colors = plt.cm.cividis(np.linspace(colormap_offset, 1, bins + 1))

    linewidth = 6.0
    num_names = len(names)

    plt.figure(figsize=(12, 6))
    for idx in range(num_names):
        name = names[idx]
        color = colors[idx]

        m = means[idx]
        plt.plot(
            m,
            color=color,
            linewidth=linewidth,
        )

        mhat = meanshat[idx]
        plt.plot(
            range(runway, mhat.shape[0] + runway),
            mhat,
            "--",
            color=color,
            linewidth=linewidth,
        )

    plt.plot([runway, runway], [1.1 * minval, 1.1 * maxval], "k--", linewidth=linewidth)

    legend_color = "0.6"
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot(
        [500, 500],
        [500, 500],
        "--",
        linewidth=linewidth,
        color=legend_color,
        label="$\hat{y}$",
    )
    plt.plot(
        [500, 500], [500, 500], linewidth=linewidth, color=legend_color, label="$y$"
    )
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.legend(fontsize=40, loc="lower right")
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Time steps", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath % "train", dpi=150)
    plt.show()
    plt.clf()

    plt.figure(figsize=(12, 6))
    for idx in range(num_names):
        name = names[idx]
        color = colors[idx]

        mtest = means_test[idx]
        plt.plot(
            mtest,
            color=color,
            linewidth=linewidth,
        )

        mhattest = means_testhat[idx]
        plt.plot(
            range(runway, mhattest.shape[0] + runway),
            mhattest,
            "--",
            color=color,
            linewidth=linewidth,
        )
    plt.plot([runway, runway], [1.1 * minval, 1.1 * maxval], "k--", linewidth=linewidth)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot(
        [500, 500],
        [500, 500],
        "--",
        linewidth=linewidth,
        color=legend_color,
        label="$\hat{y}$",
    )
    plt.plot(
        [500, 500], [500, 500], linewidth=linewidth, color=legend_color, label="$y$"
    )
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.legend(fontsize=40, loc="lower right")

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath % "test", dpi=150)
    plt.show()


def graph_stim_v_rest_deltas_3d(
    session_id,
    ch,
    session_subdir="torchraw",
    session_subdir_rest="torchbase",
    num_bins=50,
    runway=20,
    savepath=None,
):

    fig, ax = plt.subplots(figsize=(12, 12))

    means, means_stim, bin_bounds = get_stim_v_rest(
        session_id,
        ch,
        session_subdir=session_subdir,
        session_subdir_rest=session_subdir_rest,
        num_bins=num_bins,
        runway=runway,
        get_bounds=True,
    )

    # X will be initial state
    # Y will be time

    Xbins = num_bins
    Ybins = means[0].shape[0] - runway

    # We may need to modify these
    xmin = -1
    xmax = 1
    ymin = 0
    ymax = Ybins

    Z = torch.zeros(Xbins, Ybins - 1)
    for idx, m in enumerate(means):
        mstim = means_stim[idx]
        Z[idx, :] = (mstim - m)[runway:-1]
    x = [
        bin_bounds[0][0],
    ] + [b[1] for b in bin_bounds]
    # XXX this is fake but zooms in. Basically it truncates
    #  left/right so the user can see something.
    x[0] = x[1] - (x[2] - x[1])
    x[-1] = x[-2] + (x[2] - x[1])
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    colors = plt.cm.tab20b(X / X.max())
    ax.pcolormesh(y, x, Z, cmap="tab20b")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Initial state")
    if savepath:
        plt.savefig(savepath)
    plt.show()
