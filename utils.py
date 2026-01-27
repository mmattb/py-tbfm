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


class Runway:
    length = 20


class ExpEnvelope:
    decay = -0.04

    # {dtype: envelope}
    _envelopes = {}

    @classmethod
    def _refresh(cls, min_len, dtype):
        # Not sure what a faster way is...
        env = []
        for idx in range(min_len):
            env.append(math.exp(cls.decay * idx))

        cls._envelopes[dtype] = torch.tensor(env, dtype=dtype)

    @classmethod
    def get(cls, min_len, dtype):
        if dtype not in cls._envelopes:
            cls._refresh(min_len, dtype)

        env = cls._envelopes[dtype]
        if env.shape[0] < min_len:
            cls._refresh(min_len, dtype)
            env = cls._envelopes[dtype]

        return env


class PermExec(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        out = self.inner(xp)
        return out.permute(0, 2, 1)


def batch_to(batch, device):
    """
    Moves one of our complex multisession style batches to the specified device.
    """
    session_ids = batch[0]
    new_x = []
    new_stimind = []
    new_y = []

    x, si, y = batch[1:]
    for idx in range(len(session_ids)):
        new_x.append(x[idx].to(device))
        new_stimind.append(si[idx].to(device))
        new_y.append(y[idx].to(device))

    return session_ids, new_x, new_stimind, new_y


def smooth_stim_ind_exp(stim_ind, ptime1, ptime2):
    """
    stim_ind: Tensor(time, 2)
    """
    timelen = stim_ind.shape[0]
    env = ExpEnvelope.get(timelen, stim_ind.dtype)
    stim_ind[ptime1:, 0] = env[: timelen - ptime1]
    stim_ind[ptime2:, 1] = env[: timelen - ptime2]


class GetSeq(nn.Module):
    """
    Model layer which extracts the full timeline of predictions from nn.LSTM output.
    """

    def forward(self, x):
        # Output shape (batch, time, feature)
        tensor, _ = x
        return tensor


class GetSeqKeepState(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, x):
        # Output shape (batch, time, feature)
        tensor, state = x
        pred = self._model(tensor)
        return tensor, state


def get_rest_stimind(timelen, device=None):
    clock = get_clock(timelen, device=device)
    stimind = torch.zeros(timelen, 3).to(device)
    stimind[:, 2] = clock
    return stimind


def get_sp_stimind(timelen, runway=20, device=None):
    clock = get_clock(timelen, device=device)
    stimind = torch.zeros(timelen, 3).to(device)
    stimind[runway, 0] = 1.0
    stimind[:, 2] = clock
    return stimind


def eval_multistep(model, x, y, stimind_dim, tstart):
    window = x[:, :tstart, :]
    y_hat, state = model(window)
    yhats = [y_hat[:, :-1, :], y_hat[:, -1:, :]]

    timelen = x.shape[1]

    for tidx in range(tstart, timelen):
        window = torch.cat(
            [yhats[-1], x[:, tidx : tidx + 1, -1 * stimind_dim :]], axis=2
        )
        y_hat, state = model(window, state)
        yhats.append(y_hat)
    yhats = torch.cat(yhats, axis=1)

    loss = nn.MSELoss()(yhats[:, tstart:, :], y[:, tstart:, :])
    return loss.item(), yhats


def eval_multistep_multisession(model, loader, tstart, out_dir=None, prefix=""):
    batch = loader.get_batch_for_ms_graph(batch_size=10000)
    session_id, (x, stimind, y) = batch

    batch = [
        (session_id,),
        (x,),
        (stimind,),
        (y,),
    ]
    batch = batch_to(batch, model.device)

    # We are predicting out from tstart to the end of the trial.
    timelen = x.shape[1]
    pred_steps = timelen - tstart
    loss_detail, yhats, _ = model(batch, pred_steps=pred_steps, get_preds=True)
    yhats = yhats[0]

    if out_dir:
        graph_ms_progress(y, yhats, out_dir, tstart, prefix=prefix)

    return {k: v.item() for k, v in loss_detail.items()}


def get_clock(timelen, dtype=torch.float, device=None):
    return torch.arange(timelen, dtype=dtype, device=device) / timelen


def graph_ms_progress(y, yhats, out_dir, runway, prefix=""):
    meansy = torch.mean(y, axis=0).detach().cpu().numpy()
    meansyhat = torch.mean(yhats, axis=0).detach().cpu().numpy()

    # Single channel averages
    plt.plot(meansy[:, 0], label="actual")
    plt.plot(meansyhat[:, 0], label="preds")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{prefix}ms_pred0.png"))
    plt.clf()
    plt.plot(meansy[:, 30], label="actual")
    plt.plot(meansyhat[:, 30], label="preds")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{prefix}ms_pred30.png"))
    plt.clf()

    # High and low state means
    m = torch.mean(y[:, runway, 30], axis=0)
    s1 = torch.std(y[:, runway, 30], axis=0)
    s2 = 2 * s1
    high_range = [m + s1, m + s2]
    low_range = [m - s2, m - s1]

    initial_states = y[:, runway, 30]
    hidxs = torch.logical_and(
        initial_states >= high_range[0], initial_states <= high_range[1]
    )
    lidxs = torch.logical_and(
        initial_states >= low_range[0], initial_states <= low_range[1]
    )

    highs = y[hidxs, :, 30]
    lows = y[lidxs, :, 30]
    highpreds = yhats[hidxs, :, 30]
    lowpreds = yhats[lidxs, :, 30]

    highmeans = torch.mean(highs, axis=0).detach().cpu().numpy()
    lowmeans = torch.mean(lows, axis=0).detach().cpu().numpy()
    highmeanpreds = torch.mean(highpreds, axis=0).detach().cpu().numpy()
    lowmeanpreds = torch.mean(lowpreds, axis=0).detach().cpu().numpy()

    y = y.detach().cpu().numpy()
    yhats = yhats.detach().cpu().numpy()

    # Plot two actual trials and preds
    plt.plot(y[0, :, 30], label="actual")
    plt.plot(yhats[0, :, 30], label="preds")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{prefix}trial0.png"))
    plt.clf()
    plt.plot(y[1, :, 30], label="actual")
    plt.plot(yhats[1, :, 30], label="preds")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{prefix}trial1.png"))
    plt.clf()
    plt.plot(y[2, :, 30], label="actual")
    plt.plot(yhats[2, :, 30], label="preds")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{prefix}trial2.png"))
    plt.clf()

    plt.plot(highmeans, "g--", label="high state")
    plt.plot(highmeanpreds, "g")
    plt.plot(lowmeans, "r--", label="low state")
    plt.plot(lowmeanpreds, "r")
    plt.plot(meansy[:, 30], "b--", label="mean state")
    plt.plot(meansyhat[:, 30], "b")
    plt.plot([runway, runway], [-1.5, 1.5], "k--")
    plt.savefig(os.path.join(out_dir, f"{prefix}high_and_low_means.png"))
    plt.clf()


def graph_ms_ss_progress(model, x, y, out_dir, runway):
    timelen = x.shape[1]
    y_hat, _ = model(x)

    meansyh = torch.mean(y_hat, axis=0)
    means = torch.mean(y, axis=0)

    plt.plot(means[:, 30].detach().cpu().numpy(), label="actual")
    plt.plot(meansyh[:, 30].detach().cpu().numpy(), label="preds")
    plt.savefig(os.path.join(out_dir, "single_step_means.png"))
    plt.clf()

    bidx = random.randrange(y.shape[0])
    plt.plot(y[bidx, :, 30].detach().cpu().numpy(), label="actual")
    plt.plot(y_hat[bidx, :, 30].detach().cpu().numpy(), label="preds")
    plt.savefig(os.path.join(out_dir, "single_step.png"))
    plt.clf()

    loss = nn.MSELoss()(y_hat, y)
    return loss.item()


def eval_session_loss_distribution_loader(
    loader, model, pred_steps, out_dir, batch_size=10000
):
    batch = loader.get_batch_for_ms_graph(batch_size=batch_size)
    session_id, (x, stimind, y) = batch

    batch = [
        (session_id,),
        (x,),
        (stimind,),
        (y,),
    ]
    batch = batch_to(batch, model.device)
    return eval_session_loss_distribution(batch, model, pred_steps, out_dir)


def eval_session_loss_distribution(batch, model, tstart, out_dir):
    # We loop over individual sessions in the batch and get loss info.
    # Kinda slow, but doesn't require an architecture change.

    # Maps {session_id: loss_detail}
    session_losses = {}
    session_r2s = {}
    session_mean_r2s = {}
    for bidx in range(len(batch[0])):
        session_id = batch[0][bidx]
        x = batch[1][bidx]
        si = batch[2][bidx]
        y = batch[3][bidx]
        timelen = y.shape[1]
        pred_steps = timelen - tstart

        subb = ((session_id,), (x,), (si,), (y,))

        loss_detail, preds, r2s, _ = model(
            subb, pred_steps=pred_steps, get_preds=True, get_r2=True
        )
        session_losses[session_id] = loss_detail
        session_r2s[session_id] = r2s[0]

        p = torch.cat(preds, axis=0)
        meansy = torch.mean(y, axis=0)
        meansyhat = torch.mean(p, axis=0)
        session_mean_r2s[session_id] = r2_score(meansyhat, meansy)

        # TODO: remove me
        torch.save((p, y, meansyhat, meansy), os.path.join(out_dir, "sld.torch"))

        torch.save(
            (preds[0], y), os.path.join(out_dir, f"latest_testpred_{session_id}.torch")
        )

    for key in ("total", "ae", "dyn"):
        data = [ld[key].detach().cpu().numpy() for ld in session_losses.values()]
        plt.hist(data, bins=20)
        plt.savefig(os.path.join(out_dir, f"psl_{key}.png"))
        plt.clf()

    plt.hist(session_r2s.values(), bins=20)
    plt.savefig(os.path.join(out_dir, f"psl_r2.png"))
    plt.clf()

    return session_losses, session_r2s, session_mean_r2s


def plot_losses(losses, out_dir):
    plt.plot([l["total"] for l in losses], label="total", alpha=0.7)
    # plt.plot([l["ae"] for l in losses], label="ae", alpha=0.7)
    # plt.plot([l["dyn"] for l in losses], label="dyn", alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "losses.png"))
    plt.clf()


def get_preprocess_info(session_id, subdir):
    preprocess = pp.OptoStimPreprocess(session_id, base_path=subdir)
    preprocess.load_data("CondBlock1")
    ch_from_orig = preprocess.metadata["stim_Coh_from"]
    ch_to_orig = preprocess.metadata["stim_Coh_to"]
    node_position_orig = preprocess.node_position
    node_id_orig = preprocess.node_id
    preprocess.remove_bad_channels()
    preprocess.update_channel_indices()
    ch_from_after = preprocess.metadata["stim_Coh_from"]
    ch_to_after = preprocess.metadata["stim_Coh_to"]

    return (
        preprocess,
        ch_from_orig,
        ch_to_orig,
        node_position_orig,
        node_id_orig,
        ch_from_after,
        ch_to_after,
    )


def render_space(
    lfps,
    session_id,
    savefig_path=None,
    subdir="data",
    resolution=1000,
    vlims=None,
    show_legend=True,
    axs=None,
    highlight_node_id=None,
    remove_missing=False,
):
    """
    lfps: (num_ch,)
    """

    # Step 1: get session metadata
    (
        preprocess,
        ch_from_orig,
        ch_to_orig,
        node_position_orig,
        node_id_orig,
        ch_from_after,
        ch_to_after,
    ) = get_preprocess_info(session_id, subdir=subdir)

    ep = node_position_orig
    minx = 1e99
    maxx = -1e99
    maxy = -1e99
    miny = 1e99
    for epl in ep.values():
        minx = min(minx, epl[0])
        maxx = max(maxx, epl[0])
        maxy = max(maxy, epl[1])
        miny = min(miny, epl[1])

    # Make a Mahalanobis distance map for smoothing.
    buffer = 1
    weights = np.zeros((resolution, resolution, len(ep)))
    # Map orig node id -> weight at every pos index by orig node id
    mode_weights = {}
    for eidx, cidx in enumerate(ep.keys()):
        x, y = ep[cidx]
        dist = scipy.stats.multivariate_normal(mean=[x, y])

        mws = []
        for cidx2, pos2 in node_position_orig.items():
            mws.append(dist.pdf(pos2))
        mode_weights[cidx] = mws

        mg = np.meshgrid(
            np.linspace(minx - buffer, maxx + buffer, resolution),
            np.linspace(miny - buffer, maxy + buffer, resolution),
        )
        mg = np.dstack((mg[0], mg[1]))
        weights[:, :, eidx] = dist.pdf(mg)

    # Everything in this function is done in terms of the layout of the
    # actual eCog array, but we have missing electrodes.
    # node_ids maps from the LFP data onto the physical array.
    node_ids = preprocess.node_id
    # This reverses it: physical nodes to data nodes
    node_reverse = {v - 1: k for k, v in node_ids.items()}
    total_nodes = len(node_position_orig)
    interpolated = np.zeros(total_nodes)
    valid_orig_ids = set([p - 1 for p in node_ids.values()])
    for fidx in range(total_nodes):
        if fidx not in node_reverse:
            assert fidx not in valid_orig_ids
            if remove_missing:
                interpolated[fidx] = 0.0
            else:
                cur_weights = mode_weights[fidx]

                running = 0
                weight_total = 0
                for fidx_inner in range(total_nodes):
                    try:
                        translated_id = node_reverse[fidx_inner]
                    except KeyError:
                        continue
                    value = lfps[translated_id]
                    running += value * cur_weights[fidx_inner]
                    weight_total += cur_weights[fidx_inner]
                interpolated[fidx] = running / weight_total
        else:
            translated_id = node_reverse[fidx]
            interpolated[fidx] = lfps[translated_id]

    # Step 2: paint the electrodes
    if axs is None:
        plt.figure(figsize=(12, 12))
        axso = plt
        markersize = 14
    else:
        axso = axs
        markersize = 12
    for cidx, pos in node_position_orig.items():
        # 4 states: stim1, stim2, present, missing
        if cidx == ch_from_orig:
            color = "dodgerblue"
            label = "Stim 1 location"
        elif cidx == ch_to_orig:
            color = "tab:orange"
            label = "Stim 2 location"
        elif cidx in valid_orig_ids:
            color = "darkorchid"
            label = "Electrode"
        else:
            color = "dimgray"
            label = "Bad electrode"
        axso.plot(pos[0], pos[1], "o", markersize=markersize + 2, color="black")
        axso.plot(pos[0], pos[1], "o", markersize=markersize, color=color)

    if show_legend:
        p1 = matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="dodgerblue",
            label="Stim 1 location",
            linewidth=0.0,
            markersize=15,
            mec="black",
            mew=1,
        )
        p2 = matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="tab:orange",
            label="Stim 2 location",
            linewidth=0.0,
            markersize=15,
            mec="black",
            mew=1,
        )
        p3 = matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="darkorchid",
            label="Electrode",
            linewidth=0.0,
            markersize=15,
            mec="black",
            mew=1,
        )
        p4 = matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="dimgrey",
            label="Bad electrode",
            linewidth=0.0,
            markersize=15,
            mec="black",
            mew=1,
        )

        if axs is None:
            anchor = (0.5, -0.19)
        else:
            anchor = (0.5, -0.25)

        fsize_legend = 26 if axs is None else 52
        ncol = 2 if axs is None else 4
        axso.legend(
            handles=[p1, p2, p3, p4],
            bbox_to_anchor=anchor,
            loc="lower center",
            ncol=ncol,
            fontsize=fsize_legend,
        )

    axso.axis("off")

    if highlight_node_id is not None:
        ni = node_ids[highlight_node_id] - 1
        pos = node_position_orig[ni]
        axso.plot(pos[0], pos[1], "o", color="white")

    # Step 3: paint data
    mg = np.meshgrid(
        np.linspace(minx - buffer, maxx + buffer, resolution),
        np.linspace(miny - buffer, maxy + buffer, resolution),
    )
    mg = np.dstack((mg[0], mg[1]))

    denom = np.sum(weights, axis=2)

    z = np.zeros(weights.shape[:-1])
    for cidx, value in enumerate(interpolated):
        curw = weights[:, :, cidx]
        curz = curw * value.item()
        z += curz

    # Normalize
    z /= denom

    extent = (minx - buffer, maxx + buffer, miny - buffer, maxy + buffer)

    if vlims:
        vmin, vmax = vlims
    else:
        vmin = np.min(z)
        vmax = np.max(z)
    axso.imshow(z, extent=extent, vmin=vmin, vmax=vmax, origin="lower")

    if savefig_path:
        plt.tight_layout()
        plt.savefig(savefig_path, dpi=200)


def calc_r2s(yhat, y, runway=20):

    # TODO: overall score
    # TODO: this should be continuous somehow?
    pass


def split_highs_lows(yhat, y, ch=30, runway=20):
    m = torch.mean(y[:, runway, ch], axis=0)
    s = torch.std(y[:, runway, ch], axis=0)

    names = ["low", "mid-low", "mid", "mid-high", "high"]
    thresholds = [-1e99, m - s * 1.5, m - s * 0.5, m + s * 0.5, m + s * 1.5, 1e99]

    initial_states = y[:, runway, ch]
    idxs = []
    counts = []
    for ii in range(len(names)):
        cidxs = torch.logical_and(
            initial_states >= thresholds[ii], initial_states <= thresholds[ii + 1]
        )
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

    return names, means, meanshat, minval, maxval, idxs, counts


def split_highs_lows_percentiles(
    yhat, y, ch=30, runway=20, bins=7, statistic="mean"  # {'mean', 'median'}
):
    mid_idx = bins // 2
    names = []
    for idx in range(bins):
        if idx == mid_idx:
            names.append("mid")
        elif idx < mid_idx:
            names.append(f"mid-{mid_idx-idx}")
        else:
            names.append(f"mid+{idx-mid_idx}")

    initial_states = y[:, runway, ch]
    recs = [(initial_states[idx], idx) for idx in range(initial_states.shape[0])]
    recs.sort()
    rec_count = len(recs)
    bin_size = int(rec_count / bins)

    idxs = []
    counts = []
    for ii in range(len(names)):
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
        if statistic == "mean":
            m = torch.mean(y[cidxs, :, ch], axis=0).detach().cpu().numpy()
        elif statistic == "median":
            m = torch.median(y[cidxs, :, ch], axis=0).values
            m = m.detach().cpu().numpy()
        else:
            raise ValueError(
                f"Statistic must be one of 'mean', 'median'; got {statistic}"
            )

        means.append(m)
        minval = min(minval, min(m))
        maxval = max(maxval, max(m))

        if yhat is not None:
            if statistic == "mean":
                mhat = torch.mean(yhat[cidxs, :, ch], axis=0).detach().cpu().numpy()
            elif statistic == "median":
                mhat = torch.median(yhat[cidxs, :, ch], axis=0).detach().cpu().numpy()
            else:
                raise ValueError(
                    f"Statistic must be one of 'mean', 'median'; got {statistic}"
                )
            meanshat.append(mhat)
            minval = min(minval, min(m))
            maxval = max(maxval, max(m))

    return names, means, meanshat, minval, maxval, idxs, counts


def split_highs_lows_baseline(baseline, y, ch=30, runway=20):
    m = torch.mean(y[:, runway, ch], axis=0)
    s = torch.std(y[:, runway, ch], axis=0)

    names = ["low", "mid-low", "mid", "mid-high", "high"]
    thresholds = [
        m - s * 2.5,
        m - s * 1.5,
        m - s * 0.5,
        m + s * 0.5,
        m + s * 1.5,
        m + s * 2.5,
    ]

    initial_states = y[:, runway, ch]
    initial_states_base = baseline[:, runway, ch]
    idxs = []
    counts = []
    idxsbase = []
    countsbase = []
    for ii in range(len(names)):
        cidxs = torch.logical_and(
            initial_states >= thresholds[ii], initial_states <= thresholds[ii + 1]
        )
        idxs.append(cidxs)
        counts.append(len(cidxs))

        cidxsbase = torch.logical_and(
            initial_states_base >= thresholds[ii],
            initial_states_base <= thresholds[ii + 1],
        )
        idxsbase.append(cidxsbase)
        countsbase.append(len(cidxsbase))

    means = []
    meansbase = []
    minval = 1e99
    maxval = -1e99
    for iidx, cidxs in enumerate(idxs):
        m = torch.mean(y[cidxs, :, ch], axis=0).detach().cpu().numpy()
        means.append(m)
        minval = min(minval, min(m))
        maxval = max(maxval, max(m))

        cidxsb = idxsbase[iidx]
        mbase = torch.mean(baseline[cidxsb, :, ch], axis=0).detach().cpu().numpy()
        meansbase.append(mbase)
        minval = min(minval, min(m))
        maxval = max(maxval, max(m))

    return names, means, meansbase, minval, maxval, idxs, idxsbase, counts, countsbase


def graph_state_ci(
    yhat, y, statename, ch=30, runway=20, ylabel="LFP (Z-scored)", savepath=None
):
    names, means, meanshat, _, _, idxs, counts = split_highs_lows(
        yhat,
        y,
        ch=ch,
        runway=runway,
    )

    sidx = names.index(statename)
    sidx_mid = names.index("mid")
    m = means[sidx]

    if yhat is not None:
        mhat = meanshat[sidx]

    state_idxs = idxs[sidx]

    # Get per-timestep std
    std = torch.std(y[state_idxs, runway:, ch], axis=0).detach().cpu().numpy()

    mall = torch.mean(y[:, :, ch], axis=0).detach().cpu().numpy()
    stdall = torch.std(y[:, runway:, ch], axis=0).detach().cpu().numpy()
    highall = mall[runway:] + stdall
    lowall = mall[runway:] - stdall

    high = m[runway:] + std
    low = m[runway:] - std

    minval = min(min(low), min(lowall))
    maxval = max(max(high), max(highall))

    plt.figure(figsize=(12, 8))
    plt.fill_between(range(runway, m.shape[-1]), low, high, color="lightsteelblue")
    plt.plot(m, "b", label="bin $y$", linewidth=6.0)
    if yhat is not None:
        plt.plot(
            range(runway, mhat.shape[-1] + runway),
            mhat,
            "b--",
            label="$\hat{y}$",
            linewidth=6.0,
        )
    plt.plot(mall, color="darkgoldenrod", label="all $y$", linewidth=6.0)
    plt.plot([runway, runway], [1.1 * minval, 1.1 * maxval], "k--", linewidth=6.0)
    plt.fill_between(
        range(runway, m.shape[-1]), lowall, highall, color="moccasin", alpha=0.5
    )
    plt.plot(
        [
            runway,
        ],
        [
            high[0],
        ],
        "b.",
        markersize=26,
    )
    plt.plot(
        [
            runway,
        ],
        [
            low[0],
        ],
        "b.",
        markersize=26,
    )
    plt.plot(
        [
            runway,
        ],
        [
            highall[0],
        ],
        ".",
        color="darkgoldenrod",
        markersize=26,
    )
    plt.plot(
        [
            runway,
        ],
        [
            lowall[0],
        ],
        ".",
        color="darkgoldenrod",
        markersize=26,
    )
    plt.legend(fontsize=34)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Time (ms)", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


def graph_highs_lows(
    yhat,
    y,
    ch=30,
    runway=20,
    axs=None,
    legend=True,
    savepath=None,
    bins=5,
    title=None,
    colormap=None,
    colormap_offset=0.0,
    ylabel="LFP (Z-scored)",
):
    if axs is None:
        axso = plt
        linewidth = 6.0
        plt.figure(figsize=(12, 6))
    else:
        axso = axs
        linewidth = 2.0

    names, means, meanshat, minval, maxval, _, _ = split_highs_lows_percentiles(
        yhat, y, ch=ch, runway=runway, bins=bins
    )

    bins = len(names)
    if colormap:
        colors = getattr(plt.cm, colormap)(np.linspace(colormap_offset, 1, bins + 1))
    else:
        colors = plt.cm.cividis(np.linspace(colormap_offset, 1, bins + 1))
    num_names = len(names)
    for idx in range(num_names):
        m = means[idx]
        name = names[idx]
        color = colors[idx]
        axso.plot(
            m,
            color=color,
            linewidth=linewidth,
        )

        if yhat is not None:
            mhat = meanshat[idx]
            axso.plot(
                range(runway, mhat.shape[0] + runway),
                mhat,
                "--",
                color=color,
                linewidth=6.0,
            )

    legend_color = "0.6"
    xmin, xmax = axso.xlim()
    ymin, ymax = axso.ylim()
    axso.plot(
        [500, 500],
        [500, 500],
        "--",
        linewidth=6.0,
        color=legend_color,
        label="Rest",
    )
    axso.plot([500, 500], [500, 500], linewidth=6.0, color=legend_color, label="Stim")
    axso.xlim([xmin, xmax])
    axso.ylim([ymin, ymax])

    axso.plot(
        [runway, runway], [1.1 * minval, 1.1 * maxval], "k--", linewidth=linewidth
    )

    if legend:
        axso.legend(fontsize=40, loc="lower right")

    if axs is None:
        if title:
            plt.title(title, fontsize=36)

        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlabel("Time (ms)", fontsize=32)
        plt.ylabel(ylabel, fontsize=32)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=150)
        plt.show()


def calc_high_low_r2(yhat, y, runway=20, bins=5):
    num_ch = yhat.shape[-1]

    means = []
    meanshat = []
    for ch in range(num_ch):
        _, cmeans, cmeanshat, _, _, _, _ = split_highs_lows_percentiles(
            yhat, y, ch=ch, runway=runway, bins=bins
        )
        means.extend(cmeans)
        meanshat.extend(cmeanshat)

    means = torch.cat([torch.tensor(m).unsqueeze(-1) for m in means], axis=1)
    meanshat = torch.cat([torch.tensor(m).unsqueeze(-1) for m in meanshat], axis=1)

    return r2_score(meanshat.flatten(), means.flatten()).item(), means, meanshat


def graph_highs_lows_baseline(baseline, y, ch=30, runway=20):
    names, means, meansbase, minval, maxval, _, _, _, _ = split_highs_lows_baseline(
        baseline, y, ch=ch, runway=runway
    )

    colors = ["g", "r", "k", "c", "b"]
    for idx in range(len(names)):
        m = means[idx]
        name = names[idx]
        color = colors[idx]
        plt.plot(m, f"{color}--", label=f"{name}")

        mbase = meansbase[idx]
        plt.plot(mbase, color)

    plt.plot([runway, runway], [1.1 * minval, 1.1 * maxval], "k--")
    plt.legend()
    plt.show()

    return names, means, meansbase, minval, maxval


def graph_stiminds(stiminds):
    if stiminds.shape[-1] != 3:
        raise ValueError("This method supports only 3 dimensional stiminds")

    # First two dimensions are delta/1-hot funtions; 3rd is a clock vector.


def zscore_raw(data):
    mean = torch.mean(torch.mean(data, axis=0), axis=0)
    std = torch.mean(torch.std(data, axis=0), axis=0)
    return zscore(data, mean, std)


def zscore(data, mean, std):
    """
    data: (batch, time, channel)
    mean: (channel,)
    std: (channel,)
    """
    demeaned = data - mean
    zscored = demeaned / std
    return zscored
