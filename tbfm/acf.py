from collections import namedtuple
import math
import os
import random
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
import statsmodels.graphics.tsaplots
from statsmodels import tsa
import torch

Acf = namedtuple("Acf", "acf qstat pvalues")


def get_acf(x, ch):
    acf, qstat, pvalues = sm.tsa.stattools.acf(
        x[:, :, ch].flatten(), nlags=185, fft=True, qstat=True
    )
    acf = torch.tensor(acf)
    qstat = torch.tensor(qstat)
    pvalues = torch.tensor(pvalues)
    acf = Acf(acf, qstat, pvalues)
    return acf


def calc_auacf(
    t, a, horizon=None, startend=None, graph=False, weighted=False, dset_size=None
):
    auc = 0.0
    a_abs = torch.abs(a)
    tlen = len(t)
    t_norm = t / tlen

    if horizon:
        if startend:
            raise ValueError("Cannot provide both a horizon and a start-end range")

        t = t[:horizon]
        t_norm = t_norm[:horizon]
        a = a[:horizon]
        a_abs = a_abs[:horizon]
    elif startend:
        t = t[startend[0] : startend[1]]
        t_norm = t_norm[startend[0] : startend[1]]
        a = a[startend[0] : startend[1]]
        a_abs = a_abs[startend[0] : startend[1]]

    if weighted:
        if dset_size is None:
            raise ValueError("Must provide a dataset size to do weighted averaging")

        # Now our normalization is based on a weighted sum, as seen in the paper...
        ns = dset_size * tlen
        nsum = tlen * (tlen + 1) / 2
        norm_constant = 1 / (ns - nsum)
        t_norm = (dset_size - torch.arange(tlen, dtype=torch.float)) * norm_constant

    for idx, tx in enumerate(t_norm[1:]):
        if weighted:
            cacf = abs(a_abs[idx]) * tx
            auc += cacf

            if graph:
                raise NotImplementedError(
                    "Graphing with weighted averaging not implemented"
                )
        else:
            # Trapezoidal approach
            # dt = tx - t_norm[idx]
            h = min(a_abs[idx + 1], a_abs[idx])
            # auc += dt * h
            # auc += abs(a_abs[idx + 1] - a_abs[idx]) * dt / 2
            auc += a_abs[idx] / tlen

            if graph:
                mult = -1 if a[idx] < 0 else 1
                plt.fill_between(
                    [t[idx], tx * tlen], [0, 0], [mult * h, mult * h], color="b"
                )
                plt.fill_between(
                    [t[idx], tx * tlen],
                    [mult * h, mult * h],
                    [a[idx], a[idx + 1]],
                    color="b",
                )
    return auc
