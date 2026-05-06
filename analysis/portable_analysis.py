#!/usr/bin/env python3
"""
Portable TTA Cross-Validation Analysis
=======================================
Reads packaged CSV(s) and produces figures + statistics.

  Fig 1  –  Learning curve: primary TTA vs Vanilla (95 % CI and 95 % PI)
  Fig 2  –  Violin: primary TTA vs Vanilla + improvement distribution
  Fig 3  –  Cross-fold R² variability within sessions
  Fig 4  –  Headroom-corrected difficulty scatter (2×2, one panel per support size)
  Fig 5  –  CDF of headroom improvement: hard vs easy sessions
  Fig 6  –  McNemar sensitivity heatmap across R² thresholds

  When --recon-data and/or --coadapt-data are provided:
  Fig 7  –  All TTA methods learning curve (MAML, Recon, Coadapt)
  Fig 8  –  Vanilla + all TTA methods on one axis
  Fig 9  –  Violin: all TTA methods + Vanilla per support size

  P-values (per support size, for each TTA method):
    • Sign test          H0: P(TTA > Vanilla) = 0.5
    • Wilcoxon (paired)  H0: mean improvement = 0  (one-sided)
    • Brown-Forsythe     H0: var(TTA) = var(Vanilla)
    • McNemar            H0: rescued = broken

  Cross-method comparison (when multiple TTA methods provided):
    • Wilcoxon signed-rank on session means (one-sided)

Usage:
    python portable_analysis.py
    python portable_analysis.py --data tbfm_cv_data.csv \\
        --recon-data tbfm_cv_data_recon.csv \\
        --coadapt-data tbfm_cv_data_coadapt.csv \\
        --output-dir figures/

Dependencies: numpy, pandas, matplotlib, seaborn, scipy
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family":     "sans-serif",
        "font.size":       11,
        "axes.titlesize":  13,
        "axes.labelsize":  12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi":      150,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    }
)

TTA_COLOR     = "#2196F3"   # blue  (MAML norecon)
VAN_COLOR     = "#FF5722"   # orange-red  (Vanilla)
DIFF_COLOR    = "#43A047"   # green
RECON_COLOR   = "#FF9800"   # orange  (MAML recon)
COADAPT_COLOR = "#9C27B0"   # purple  (Coadapt)

METHOD_STYLES = {
    "MAML":    dict(color=TTA_COLOR,     marker="o", ls="-",  label="MAML TTA"),
    "Recon":   dict(color=RECON_COLOR,   marker="D", ls="-",  label="Recon TTA"),
    "Coadapt": dict(color=COADAPT_COLOR, marker="^", ls="-",  label="Coadapt TTA"),
    "Vanilla": dict(color=VAN_COLOR,     marker="s", ls="--", label="Vanilla"),
}


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "session_id", "fold", "support_samples",
        "tta_r2", "vanilla_test_r2", "vanilla_train_r2",
        "improvement_vs_vanilla",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────
def compute_session_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average TTA R² across folds for each (session, support_size)."""
    g = df.groupby(["session_id", "support_samples"])
    out = g.agg(
        tta_mean   = ("tta_r2",          "mean"),
        tta_std    = ("tta_r2",          "std"),
        n_folds    = ("tta_r2",          "count"),
        vanilla_r2 = ("vanilla_test_r2", "first"),
        monkey     = ("monkey",          "first"),
        area       = ("area",            "first"),
    ).reset_index()
    out["improvement"] = out["tta_mean"] - out["vanilla_r2"]
    return out


def compute_lc_stats(smeans: pd.DataFrame) -> pd.DataFrame:
    """Per support size: mean, 95 % CI, and 95 % PI."""
    rows = []
    for ss, g in smeans.groupby("support_samples"):
        tta = g["tta_mean"].values
        van = g["vanilla_r2"].values
        n   = len(tta)
        tc  = stats.t.ppf(0.975, df=n - 1)

        def _stats(x):
            mu = x.mean()
            se = x.std(ddof=1) / np.sqrt(n)
            sd = x.std(ddof=1)
            return (
                mu,
                mu - tc * se,  mu + tc * se,
                mu - tc * sd * np.sqrt(1 + 1 / n),
                mu + tc * sd * np.sqrt(1 + 1 / n),
            )

        t_mu, t_ci_lo, t_ci_hi, t_pi_lo, t_pi_hi = _stats(tta)
        v_mu, v_ci_lo, v_ci_hi, v_pi_lo, v_pi_hi = _stats(van)

        rows.append({
            "support_samples": ss, "n": n,
            "tta_mean":  t_mu,
            "tta_ci_lo": t_ci_lo, "tta_ci_hi": t_ci_hi,
            "tta_pi_lo": t_pi_lo, "tta_pi_hi": t_pi_hi,
            "van_mean":  v_mu,
            "van_ci_lo": v_ci_lo, "van_ci_hi": v_ci_hi,
            "van_pi_lo": v_pi_lo, "van_pi_hi": v_pi_hi,
        })
    return pd.DataFrame(rows).sort_values("support_samples").reset_index(drop=True)


# ── Figure 1: Learning curve ──────────────────────────────────────────────────
def fig1_learning_curve(lc: pd.DataFrame, out: Path,
                         label: str = "Recon TTA", color: str = RECON_COLOR) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x      = np.arange(len(lc))
    labels = lc["support_samples"].astype(str).tolist()

    ax.fill_between(x, lc["tta_pi_lo"], lc["tta_pi_hi"],
                    color=color, alpha=0.15, label=f"{label} 95 % PI")
    ax.errorbar(x, lc["tta_mean"],
                yerr=[lc["tta_mean"] - lc["tta_ci_lo"],
                      lc["tta_ci_hi"] - lc["tta_mean"]],
                fmt="o-", color=color, lw=2, ms=7,
                capsize=5, capthick=1.5, elinewidth=1.5,
                label=f"{label} mean ± 95 % CI")

    ax.fill_between(x, lc["van_pi_lo"], lc["van_pi_hi"],
                    color=VAN_COLOR, alpha=0.15, label="Vanilla 95 % PI")
    ax.errorbar(x, lc["van_mean"],
                yerr=[lc["van_mean"] - lc["van_ci_lo"],
                      lc["van_ci_hi"] - lc["van_mean"]],
                fmt="s--", color=VAN_COLOR, lw=2, ms=7,
                capsize=5, capthick=1.5, elinewidth=1.5,
                label="Vanilla mean ± 95 % CI")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Support samples")
    ax.set_ylabel("R²")
    ax.set_title(f"Fig 1  ·  Learning curve with CI and PI  ({label})")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out / "fig1_learning_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Violin plot ─────────────────────────────────────────────────────
def fig2_violin(smeans: pd.DataFrame, out: Path, label: str = "TTA") -> None:
    tta_df = smeans[["session_id", "support_samples", "tta_mean"]].copy()
    tta_df = tta_df.rename(columns={"tta_mean": "r2"})
    tta_df["method"] = label

    van_df = smeans[["session_id", "support_samples", "vanilla_r2"]].copy()
    van_df = van_df.rename(columns={"vanilla_r2": "r2"})
    van_df["method"] = "Vanilla"

    long = pd.concat([tta_df, van_df], ignore_index=True)
    long["support_samples"] = long["support_samples"].astype(str)
    support_order = [str(s) for s in sorted(smeans["support_samples"].unique())]

    impr = smeans[["support_samples", "improvement"]].copy()
    impr["support_samples"] = impr["support_samples"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tta_color = METHOD_STYLES.get(label, METHOD_STYLES["MAML"])["color"]
    palette = {label: tta_color, "Vanilla": VAN_COLOR}
    sns.violinplot(
        data=long, x="support_samples", y="r2", hue="method",
        split=True, inner="quart", palette=palette,
        order=support_order, ax=axes[0], linewidth=1.2,
    )
    axes[0].set_title(f"R² distributions: {label} vs Vanilla")
    axes[0].set_xlabel("Support samples")
    axes[0].set_ylabel("R²")
    axes[0].legend(title=None)
    axes[0].grid(True, axis="y", alpha=0.35)

    sns.violinplot(
        data=impr, x="support_samples", y="improvement",
        color=DIFF_COLOR, inner="quart", order=support_order,
        ax=axes[1], linewidth=1.2,
    )
    axes[1].axhline(0, color="k", lw=1.5, linestyle="--", label="No change")
    axes[1].set_title(f"Improvement: {label} − Vanilla")
    axes[1].set_xlabel("Support samples")
    axes[1].set_ylabel("ΔR²")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.35)

    fig.suptitle(f"Fig 2  ·  Violin plots  ({label})", fontsize=14)
    fig.tight_layout()
    path = out / "fig2_violin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Within-session cross-fold variability ───────────────────────────
def fig3_fold_variability(df: pd.DataFrame, out: Path, label: str = "TTA") -> None:
    rows = []
    for (sid, ss), g in df.groupby(["session_id", "support_samples"]):
        r2s = g["tta_r2"].values
        if len(r2s) < 2:
            continue
        mu = r2s.mean()
        for r2 in r2s:
            rows.append({"session_id": sid, "support_samples": str(ss), "deviation": r2 - mu})
    dev = pd.DataFrame(rows)
    support_order = [str(s) for s in sorted(df["support_samples"].unique())]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(
        data=dev, x="support_samples", y="deviation",
        color=TTA_COLOR, inner="quart", ax=ax,
        order=support_order, linewidth=1.2,
    )
    ax.axhline(0, color="k", lw=1.5, linestyle="--")
    ax.set_xlabel("Support samples")
    ax.set_ylabel("R² deviation from session mean")
    ax.set_title(f"Fig 3  ·  Cross-fold variability within sessions  ({label})")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    path = out / "fig3_fold_variability.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Difficulty scatter ──────────────────────────────────────────────
AREA_COLORS = {"S1": TTA_COLOR, "M1": VAN_COLOR}


def compute_headroom(smeans: pd.DataFrame, cap: float = 0.95) -> pd.DataFrame:
    df = smeans.copy()
    excluded = (df["vanilla_r2"] >= cap).sum()
    if excluded:
        print(f"  Headroom: excluding {excluded} rows with vanilla R² ≥ {cap}")
    df = df[df["vanilla_r2"] < cap].copy()
    df["headroom_improvement"] = (
        (df["tta_mean"] - df["vanilla_r2"]) / (1 - df["vanilla_r2"])
    )
    return df


def compute_headroom_stats(smeans: pd.DataFrame, cap: float = 0.95) -> pd.DataFrame:
    hdf  = compute_headroom(smeans, cap=cap)
    rows = []
    for ss, g in hdf.groupby("support_samples"):
        van = g["vanilla_r2"].values
        hi  = g["headroom_improvement"].values
        rho, sp_p = stats.spearmanr(van, hi, alternative="less")
        med   = np.median(van)
        hard  = hi[van <= med]
        easy  = hi[van >  med]
        mw_u, mw_p = stats.mannwhitneyu(hard, easy, alternative="greater")
        rows.append({
            "support_samples": ss,
            "n":               len(g),
            "n_hard":          len(hard),
            "n_easy":          len(easy),
            "spearman_rho":    rho,
            "spearman_p":      sp_p,
            "mw_U":            mw_u,
            "mw_p":            mw_p,
        })
    return pd.DataFrame(rows)


def fig4_difficulty_scatter(smeans: pd.DataFrame, out: Path, cap: float = 0.95) -> None:
    hdf = compute_headroom(smeans, cap=cap)
    support_sizes = sorted(hdf["support_samples"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, ss in zip(axes, support_sizes):
        g = hdf[hdf["support_samples"] == ss]
        for area, sub in g.groupby("area"):
            ax.scatter(sub["vanilla_r2"], sub["headroom_improvement"],
                       color=AREA_COLORS.get(area, "gray"),
                       label=area, alpha=0.75, s=40, zorder=3)
        x = g["vanilla_r2"].values
        y = g["headroom_improvement"].values
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, color="k", lw=1.2, linestyle="--", zorder=2)
        rho, sp_p = stats.spearmanr(x, y, alternative="less")

        def _sig(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        ax.text(
            0.97, 0.97,
            f"Spearman ρ = {rho:+.2f}  {_sig(sp_p)}\np = {sp_p:.3g}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85),
        )
        ax.axhline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_title(f"Support = {ss}")
        ax.set_xlabel("Vanilla R²")
        ax.set_ylabel("Headroom improvement")
        ax.legend(fontsize=8, title="Area", title_fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Fig 4  ·  TTA benefit vs session difficulty (ceiling-corrected)\n"
        r"Headroom improvement $= \frac{R^2_\mathrm{TTA} - R^2_\mathrm{van}}"
        r"{1 - R^2_\mathrm{van}}$",
        fontsize=12,
    )
    fig.tight_layout()
    path = out / "fig4_difficulty_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def print_headroom_stats(hs: pd.DataFrame, label: str = "TTA") -> None:
    rule = "=" * 72
    print()
    print(rule)
    print(f"  DIFFICULTY ANALYSIS  [{label}]  (headroom-corrected, vanilla R² split at median)")
    print("  Spearman H1: ρ < 0  (harder → more benefit)")
    print("  Mann-Whitney H1: hard sessions > easy sessions  (one-sided)")
    print(rule)
    print(
        f"{'Support':>8}  {'n':>4}  "
        f"{'ρ':>7}  {'Spear p':>9}  "
        f"{'MW U':>7}  {'MW p':>9}  {'sig':>4}"
    )
    print("-" * 72)
    for _, row in hs.iterrows():
        def _sig(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        worst_p = max(row["spearman_p"], row["mw_p"])
        print(
            f"{int(row['support_samples']):>8}  "
            f"{int(row['n']):>4}  "
            f"{row['spearman_rho']:>+7.3f}  "
            f"{row['spearman_p']:>9.3g}  "
            f"{int(row['mw_U']):>7}  "
            f"{row['mw_p']:>9.3g}  "
            f"{_sig(worst_p):>4}"
        )
    print(rule)


# ── Figure 5: CDF of headroom improvement ────────────────────────────────────
HARD_COLOR = "#C62828"
EASY_COLOR = "#2E7D32"


def fig5_cdf(smeans: pd.DataFrame, out: Path, cap: float = 0.95) -> None:
    hdf           = compute_headroom(smeans, cap=cap)
    support_sizes = sorted(hdf["support_samples"].unique())

    def ecdf_vals(x):
        xs = np.sort(x)
        ys = np.arange(1, len(x) + 1) / len(x)
        return xs, ys

    def ecdf_at(sorted_x, x_grid):
        return np.searchsorted(sorted_x, x_grid, side="right") / len(sorted_x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    for ax, ss in zip(axes, support_sizes):
        g   = hdf[hdf["support_samples"] == ss]
        van = g["vanilla_r2"].values
        hi  = g["headroom_improvement"].values

        med  = np.median(van)
        hard = np.sort(hi[van <= med])
        easy = np.sort(hi[van >  med])

        hx, hy = ecdf_vals(hard)
        ex, ey = ecdf_vals(easy)

        x_lo   = min(hx[0],  ex[0])  - 0.05
        x_hi   = max(hx[-1], ex[-1]) + 0.05
        x_grid = np.linspace(x_lo, x_hi, 1000)
        F_hard = ecdf_at(hard, x_grid)
        F_easy = ecdf_at(easy, x_grid)

        ax.fill_between(x_grid, F_hard, F_easy,
                        where=(F_easy >= F_hard),
                        color="gray", alpha=0.20, zorder=1)
        ax.step(np.r_[x_lo, hx], np.r_[0, hy],
                where="post", color=HARD_COLOR, lw=2, label=f"Hard  (n={len(hard)})")
        ax.step(np.r_[x_lo, ex], np.r_[0, ey],
                where="post", color=EASY_COLOR, lw=2, label=f"Easy  (n={len(easy)})")

        mw_u, mw_p = stats.mannwhitneyu(hard, easy, alternative="greater")
        p_sup = mw_u / (len(hard) * len(easy))
        sig   = "***" if mw_p < 0.001 else "**" if mw_p < 0.01 else "*" if mw_p < 0.05 else "ns"
        ax.text(
            0.03, 0.97,
            f"P̂(hard > easy) = {p_sup:.2f}\np = {mw_p:.3g}  {sig}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85),
        )

        ax.axvline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_title(f"Support = {ss}")
        ax.set_xlabel("Headroom improvement")
        ax.set_ylabel("Cumulative probability")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.30)

    fig.suptitle(
        "Fig 5  ·  CDF of headroom improvement: hard vs easy sessions\n"
        "(split at median vanilla R²;  shaded area = CDF separation)",
        fontsize=12,
    )
    fig.tight_layout()
    path = out / "fig5_cdf.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── McNemar sensitivity ───────────────────────────────────────────────────────
POOR_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def compute_mcnemar_sensitivity(smeans: pd.DataFrame,
                                 thresholds: list = POOR_THRESHOLDS) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        for ss, g in smeans.groupby("support_samples"):
            tta = g["tta_mean"].values
            van = g["vanilla_r2"].values
            n   = len(tta)

            van_poor = van < thr
            tta_poor = tta < thr
            n_rescued    = int((van_poor & ~tta_poor).sum())
            n_broken     = int((~van_poor & tta_poor).sum())
            n_discordant = n_rescued + n_broken

            if n_discordant > 0:
                p = stats.binomtest(n_rescued, n_discordant, p=0.5, alternative="greater").pvalue
            else:
                p = np.nan

            rows.append({
                "threshold":       thr,
                "support_samples": ss,
                "n":               n,
                "n_van_poor":      int(van_poor.sum()),
                "n_tta_poor":      int(tta_poor.sum()),
                "n_rescued":       n_rescued,
                "n_broken":        n_broken,
                "mcnemar_p":       p,
            })
    return pd.DataFrame(rows)


def fig6_mcnemar_heatmap(sens: pd.DataFrame, out: Path) -> None:
    thresholds    = sorted(sens["threshold"].unique())
    support_sizes = sorted(sens["support_samples"].unique())

    pmat = np.full((len(thresholds), len(support_sizes)), np.nan)
    ann  = np.empty((len(thresholds), len(support_sizes)), dtype=object)

    for i, thr in enumerate(thresholds):
        for j, ss in enumerate(support_sizes):
            row = sens[(sens["threshold"] == thr) & (sens["support_samples"] == ss)].iloc[0]
            p   = row["mcnemar_p"]
            pmat[i, j] = -np.log10(p) if (p > 0 and not np.isnan(p)) else np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ann[i, j]  = (f"{int(row['n_rescued'])} / {int(row['n_broken'])}\n"
                          f"p={p:.2g} {sig}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pmat, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(6, np.nanmax(pmat)))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("−log₁₀(McNemar p)", fontsize=10)
    cbar.ax.axhline(-np.log10(0.05), color="k", lw=1.5, linestyle="--")
    cbar.ax.text(1.05, -np.log10(0.05) / max(6, np.nanmax(pmat)),
                 "p=0.05", transform=cbar.ax.transAxes, va="center", fontsize=8)

    for i in range(len(thresholds)):
        for j in range(len(support_sizes)):
            ax.text(j, i, ann[i, j], ha="center", va="center",
                    fontsize=8, family="monospace",
                    color="white" if pmat[i, j] > 3 else "black")

    ax.set_xticks(range(len(support_sizes)))
    ax.set_xticklabels([str(s) for s in support_sizes])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"R² < {t}" for t in thresholds])
    ax.set_xlabel("Support samples")
    ax.set_title(
        "Fig 6  ·  McNemar sensitivity: rescued / broken sessions\n"
        "across R² thresholds and support sizes"
    )
    fig.tight_layout()
    path = out / "fig6_mcnemar_sensitivity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def print_mcnemar_sensitivity(sens: pd.DataFrame, label: str = "TTA") -> None:
    rule = "=" * 76
    print()
    print(rule)
    print(f"  McNEMAR SENSITIVITY  [{label}]  (rescued / broken  across R² thresholds)")
    print("  H1: TTA rescues more sessions than it breaks  (one-sided)")
    print(rule)
    for thr, grp in sens.groupby("threshold"):
        print(f"\n  Threshold R² < {thr}")
        print(f"  {'Support':>8}  {'Vanilla':>9}  {'TTA':>9}  "
              f"{'Rescued':>8}  {'Broken':>7}  {'p':>10}  {'sig':>4}")
        print(f"  {'-'*62}")
        for _, row in grp.iterrows():
            n   = int(row["n"])
            p   = row["mcnemar_p"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(
                f"  {int(row['support_samples']):>8}  "
                f"  {int(row['n_van_poor']):>2}/{n} ({row['n_van_poor']/n:>3.0%})  "
                f"  {int(row['n_tta_poor']):>2}/{n} ({row['n_tta_poor']/n:>3.0%})  "
                f"  {int(row['n_rescued']):>5}  "
                f"  {int(row['n_broken']):>5}  "
                f"  {p:>10.3g}  {sig:>4}"
            )
    print()
    print(rule)


# ── P-value table ─────────────────────────────────────────────────────────────
def compute_pvalues(smeans: pd.DataFrame, poor_threshold: float = 0.05) -> pd.DataFrame:
    rows = []
    for ss, g in smeans.groupby("support_samples"):
        tta  = g["tta_mean"].values
        van  = g["vanilla_r2"].values
        diff = tta - van
        n    = len(diff)

        n_wins   = int((diff > 0).sum())
        win_rate = n_wins / n
        binom_p  = stats.binomtest(n_wins, n, p=0.5, alternative="greater").pvalue

        if np.all(diff == 0):
            w_stat, w_p = np.nan, np.nan
        else:
            w_res = stats.wilcoxon(diff, alternative="greater")
            w_stat, w_p = w_res.statistic, w_res.pvalue

        bf_stat, bf_p_two = stats.levene(tta, van, center="median")
        bf_p = bf_p_two / 2 if van.std() > tta.std() else 1 - bf_p_two / 2

        van_poor = van < poor_threshold
        tta_poor = tta < poor_threshold
        n_van_poor = int(van_poor.sum())
        n_tta_poor = int(tta_poor.sum())
        n_rescued    = int((van_poor & ~tta_poor).sum())
        n_broken     = int((~van_poor & tta_poor).sum())
        n_discordant = n_rescued + n_broken

        if n_discordant > 0:
            mcnemar_p = stats.binomtest(
                n_rescued, n_discordant, p=0.5, alternative="greater"
            ).pvalue
        else:
            mcnemar_p = np.nan

        rows.append({
            "support_samples": ss,
            "n_sessions":      n,
            "n_wins":          n_wins,
            "win_rate":        win_rate,
            "win_rate_p":      binom_p,
            "wilcoxon_W":      w_stat,
            "wilcoxon_p":      w_p,
            "BF_stat":         bf_stat,
            "BF_p":            bf_p,
            "n_tta_poor":      n_tta_poor,
            "pct_tta_poor":    n_tta_poor / n,
            "n_van_poor":      n_van_poor,
            "pct_van_poor":    n_van_poor / n,
            "n_rescued":       n_rescued,
            "n_broken":        n_broken,
            "mcnemar_p":       mcnemar_p,
            "poor_threshold":  poor_threshold,
        })
    return pd.DataFrame(rows)


def print_pvalues(pv: pd.DataFrame, label: str = "TTA") -> None:
    rule = "=" * 68
    print()
    print(rule)
    print(f"  P-VALUE SUMMARY  [{label} vs Vanilla]")
    print(rule)
    hdr = (
        f"{'Support':>8}  {'n':>4}  "
        f"{'Win%':>6}  {'Binom p':>10}  "
        f"{'Wilcox p':>10}  {'BF p':>10}"
    )
    print(hdr)
    print("-" * 68)
    for _, row in pv.iterrows():
        print(
            f"{int(row['support_samples']):>8}  "
            f"{int(row['n_sessions']):>4}  "
            f"{row['win_rate']:>5.1%}  "
            f"{row['win_rate_p']:>10.3g}  "
            f"{row['wilcoxon_p']:>10.3g}  "
            f"{row['BF_p']:>10.3g}"
        )
    print(rule)
    print("  Win%     = fraction of sessions where TTA R² > Vanilla R²")
    print("  Binom p  = sign test H0: P(win)=50%  (one-sided, >)")
    print("  Wilcox p = Wilcoxon signed-rank H0: improvement=0  (one-sided, >)")
    print("  BF p     = Brown-Forsythe H0: var(TTA)=var(Vanilla)  H1: var(van)>var(TTA)  (one-sided)")
    print(rule)

    thr  = pv["poor_threshold"].iloc[0]
    rule2 = "=" * 72
    print()
    print(rule2)
    print(f"  POOR-FIT SESSIONS  [{label}]  (R² < {thr})")
    print(f"  McNemar H1: TTA rescues more sessions than it breaks  (one-sided)")
    print(rule2)
    print(
        f"  {'Support':>8}  {'Vanilla':>10}  {'TTA':>10}  "
        f"{'Rescued':>9}  {'Broken':>7}  {'McNemar p':>11}"
    )
    print(f"  {'-'*68}")
    for _, row in pv.iterrows():
        def _sig(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        n  = int(row["n_sessions"])
        p  = row["mcnemar_p"]
        print(
            f"  {int(row['support_samples']):>8}  "
            f"  {int(row['n_van_poor']):>2}/{n} ({row['pct_van_poor']:>3.0%})  "
            f"  {int(row['n_tta_poor']):>2}/{n} ({row['pct_tta_poor']:>3.0%})  "
            f"  {int(row['n_rescued']):>4} / {int(row['n_broken']):<4}  "
            f"  {p:>8.3g} {_sig(p)}"
        )
    print(rule2)
    print("  Rescued = vanilla poor → TTA good")
    print("  Broken  = vanilla good → TTA poor")
    print(rule2)


# ── Cross-method comparison stats ─────────────────────────────────────────────
def compute_crossmethod_stats(smeans_a: pd.DataFrame, smeans_b: pd.DataFrame,
                               label_a: str, label_b: str) -> pd.DataFrame:
    """
    Paired Wilcoxon on session means for each support size.
    H1: label_a > label_b  (one-sided).
    Uses inner join on session_id — compares only sessions present in both.
    """
    rows = []
    for ss in sorted(set(smeans_a["support_samples"]) & set(smeans_b["support_samples"])):
        a = smeans_a[smeans_a["support_samples"] == ss].set_index("session_id")["tta_mean"]
        b = smeans_b[smeans_b["support_samples"] == ss].set_index("session_id")["tta_mean"]
        common = a.index.intersection(b.index)
        a, b = a.loc[common].values, b.loc[common].values
        diff = a - b
        n = len(diff)
        n_wins = int((diff > 0).sum())
        if np.all(diff == 0):
            w_stat, w_p = np.nan, np.nan
        else:
            w_res = stats.wilcoxon(diff, alternative="greater")
            w_stat, w_p = w_res.statistic, w_res.pvalue
        rows.append({
            "support_samples": ss,
            "n": n,
            "n_wins": n_wins,
            "win_rate": n_wins / n,
            "wilcoxon_W": w_stat,
            "wilcoxon_p": w_p,
        })
    return pd.DataFrame(rows)


def print_crossmethod_stats(df: pd.DataFrame, label_a: str, label_b: str) -> None:
    rule = "=" * 68
    print()
    print(rule)
    print(f"  CROSS-METHOD COMPARISON  [{label_a} vs {label_b}]")
    print(f"  H1: {label_a} > {label_b}  (Wilcoxon on session means, one-sided)")
    print(rule)
    print(f"{'Support':>8}  {'n':>4}  {'Win%':>6}  {'Wilcox p':>10}  {'sig':>4}")
    print("-" * 68)

    def _sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    for _, row in df.iterrows():
        p = row["wilcoxon_p"]
        print(
            f"{int(row['support_samples']):>8}  "
            f"{int(row['n']):>4}  "
            f"{row['win_rate']:>5.1%}  "
            f"{p:>10.3g}  "
            f"{_sig(p):>4}"
        )
    print(rule)


# ── Figures 7–9: multi-method comparisons ─────────────────────────────────────
def _align_lc(lc_dict: dict[str, pd.DataFrame]):
    """Intersect support sizes across all methods; return aligned subset."""
    shared = set.intersection(*[set(lc["support_samples"]) for lc in lc_dict.values()])
    aligned = {}
    for name, lc in lc_dict.items():
        aligned[name] = (
            lc[lc["support_samples"].isin(shared)]
            .sort_values("support_samples")
            .reset_index(drop=True)
        )
    x      = np.arange(len(next(iter(aligned.values()))))
    labels = next(iter(aligned.values()))["support_samples"].astype(str).tolist()
    return aligned, x, labels


def fig7_tta_comparison(lc_dict: dict[str, pd.DataFrame], out: Path) -> None:
    """All TTA methods on one axis (no vanilla)."""
    aligned, x, labels = _align_lc(lc_dict)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, lc in aligned.items():
        style = METHOD_STYLES.get(name, dict(color="gray", marker="o", ls="-", label=name))
        ax.fill_between(x, lc["tta_pi_lo"], lc["tta_pi_hi"],
                        color=style["color"], alpha=0.10)
        ax.errorbar(x, lc["tta_mean"],
                    yerr=[lc["tta_mean"] - lc["tta_ci_lo"],
                          lc["tta_ci_hi"] - lc["tta_mean"]],
                    fmt=f"{style['marker']}{style['ls']}",
                    color=style["color"], lw=2, ms=7,
                    capsize=5, capthick=1.5, elinewidth=1.5,
                    label=f"{style['label']} mean ± 95 % CI")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Support samples")
    ax.set_ylabel("R²")
    ax.set_title("Fig 7  ·  All TTA methods learning curve")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out / "fig7_tta_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig8_all_methods(lc_dict: dict[str, pd.DataFrame], lc_primary: pd.DataFrame,
                     out: Path) -> None:
    """Vanilla + all TTA methods on one axis."""
    aligned, x, labels = _align_lc(lc_dict)
    lc_van = lc_primary[lc_primary["support_samples"].isin(
        next(iter(aligned.values()))["support_samples"]
    )].sort_values("support_samples").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(x, lc_van["van_pi_lo"], lc_van["van_pi_hi"],
                    color=VAN_COLOR, alpha=0.10)
    ax.errorbar(x, lc_van["van_mean"],
                yerr=[lc_van["van_mean"] - lc_van["van_ci_lo"],
                      lc_van["van_ci_hi"] - lc_van["van_mean"]],
                fmt="s--", color=VAN_COLOR, lw=2, ms=7,
                capsize=5, capthick=1.5, elinewidth=1.5,
                label="Vanilla mean ± 95 % CI")

    for name, lc in aligned.items():
        style = METHOD_STYLES.get(name, dict(color="gray", marker="o", ls="-", label=name))
        ax.fill_between(x, lc["tta_pi_lo"], lc["tta_pi_hi"],
                        color=style["color"], alpha=0.10)
        ax.errorbar(x, lc["tta_mean"],
                    yerr=[lc["tta_mean"] - lc["tta_ci_lo"],
                          lc["tta_ci_hi"] - lc["tta_mean"]],
                    fmt=f"{style['marker']}{style['ls']}",
                    color=style["color"], lw=2, ms=7,
                    capsize=5, capthick=1.5, elinewidth=1.5,
                    label=f"{style['label']} mean ± 95 % CI")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Support samples")
    ax.set_ylabel("R²")
    ax.set_title("Fig 8  ·  Vanilla + all TTA methods")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out / "fig8_all_methods.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig9_comparison_violin(smeans_dict: dict[str, pd.DataFrame], out: Path) -> None:
    """Violin of session-mean R² for all TTA methods + vanilla, per support size."""
    # Build long-form dataframe
    rows = []
    for name, sm in smeans_dict.items():
        for _, row in sm.iterrows():
            rows.append({"method": name, "support_samples": str(int(row["support_samples"])),
                         "r2": row["tta_mean"]})
        # Add vanilla from the first method only
    van_sm = next(iter(smeans_dict.values()))
    for _, row in van_sm.iterrows():
        rows.append({"method": "Vanilla", "support_samples": str(int(row["support_samples"])),
                     "r2": row["vanilla_r2"]})

    long = pd.DataFrame(rows)
    support_order = [str(s) for s in sorted(van_sm["support_samples"].unique())]
    method_order  = list(smeans_dict.keys()) + ["Vanilla"]
    palette = {
        **{k: METHOD_STYLES.get(k, {}).get("color", "gray") for k in smeans_dict},
        "Vanilla": VAN_COLOR,
    }

    fig, axes = plt.subplots(1, len(support_order), figsize=(4 * len(support_order), 5),
                              sharey=True)
    if len(support_order) == 1:
        axes = [axes]

    for ax, ss in zip(axes, support_order):
        sub = long[long["support_samples"] == ss]
        sns.violinplot(
            data=sub, x="method", y="r2", hue="method",
            order=method_order, hue_order=method_order,
            palette=palette, legend=False,
            inner="quart", ax=ax, linewidth=1.2,
        )
        ax.set_title(f"Support = {ss}")
        ax.set_xlabel("")
        ax.set_ylabel("R²" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.35)
        ax.axhline(0, color="k", lw=0.5, linestyle=":")

    fig.suptitle("Fig 9  ·  R² distributions: all methods per support size", fontsize=14)
    fig.tight_layout()
    path = out / "fig9_comparison_violin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portable TTA analysis: figures + p-values"
    )
    parser.add_argument("--data", default="tbfm_cv_data.csv",
                        help="Primary TTA CSV — MAML norecon  (default: tbfm_cv_data.csv)")
    parser.add_argument("--recon-data", default=None,
                        help="Path to recon MAML CSV  (tbfm_cv_data_recon.csv)")
    parser.add_argument("--coadapt-data", default=None,
                        help="Path to coadapt CSV  (tbfm_cv_data_coadapt.csv)")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory for output PNGs  (default: figures/)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load primary (MAML norecon) ────────────────────────────────────────────
    data_path = Path(args.data)
    print(f"Loading primary data from {data_path} …")
    df = load_data(data_path)
    print(
        f"  {len(df)} rows  |  {df['session_id'].nunique()} sessions  |  "
        f"{df['fold'].nunique()} folds  |  support: {sorted(df['support_samples'].unique())}"
    )

    smeans = compute_session_means(df)
    lc     = compute_lc_stats(smeans)

    print("\nGenerating Figs 1–6 (MAML vs Vanilla) …")
    fig1_learning_curve(lc, out_dir, label="MAML TTA")
    fig2_violin(smeans, out_dir, label="MAML")
    fig3_fold_variability(df, out_dir, label="MAML")

    pv = compute_pvalues(smeans)
    print_pvalues(pv, label="MAML")
    pv.to_csv(out_dir / "pvalues_maml.csv", index=False)
    print(f"  P-value table → {out_dir}/pvalues_maml.csv")

    hs = compute_headroom_stats(smeans)
    fig4_difficulty_scatter(smeans, out_dir)
    fig5_cdf(smeans, out_dir)
    print_headroom_stats(hs, label="MAML")
    hs.to_csv(out_dir / "headroom_stats_maml.csv", index=False)
    print(f"  Headroom stats → {out_dir}/headroom_stats_maml.csv")

    sens = compute_mcnemar_sensitivity(smeans)
    fig6_mcnemar_heatmap(sens, out_dir)
    print_mcnemar_sensitivity(sens, label="MAML")
    sens.to_csv(out_dir / "mcnemar_sensitivity_maml.csv", index=False)
    print(f"  McNemar sensitivity → {out_dir}/mcnemar_sensitivity_maml.csv")

    # ── Optional: recon ────────────────────────────────────────────────────────
    smeans_recon = lc_recon = None
    if args.recon_data:
        recon_path = Path(args.recon_data)
        print(f"\nLoading recon data from {recon_path} …")
        df_recon     = load_data(recon_path)
        smeans_recon = compute_session_means(df_recon)
        lc_recon     = compute_lc_stats(smeans_recon)
        print(
            f"  {len(df_recon)} rows  |  {df_recon['session_id'].nunique()} sessions  |  "
            f"{df_recon['fold'].nunique()} folds  |  support: {sorted(df_recon['support_samples'].unique())}"
        )
        print("\nGenerating recon stats …")
        pv_recon = compute_pvalues(smeans_recon)
        print_pvalues(pv_recon, label="Recon")
        pv_recon.to_csv(out_dir / "pvalues_recon.csv", index=False)
        print(f"  P-value table → {out_dir}/pvalues_recon.csv")

        hs_recon = compute_headroom_stats(smeans_recon)
        print_headroom_stats(hs_recon, label="Recon")
        hs_recon.to_csv(out_dir / "headroom_stats_recon.csv", index=False)

        sens_recon = compute_mcnemar_sensitivity(smeans_recon)
        print_mcnemar_sensitivity(sens_recon, label="Recon")
        sens_recon.to_csv(out_dir / "mcnemar_sensitivity_recon.csv", index=False)

        cm = compute_crossmethod_stats(smeans_recon, smeans, "Recon", "MAML")
        print_crossmethod_stats(cm, "Recon", "MAML")
        cm.to_csv(out_dir / "crossmethod_recon_vs_maml.csv", index=False)

    # ── Optional: coadapt ──────────────────────────────────────────────────────
    smeans_coadapt = lc_coadapt = None
    if args.coadapt_data:
        coadapt_path = Path(args.coadapt_data)
        print(f"\nLoading coadapt data from {coadapt_path} …")
        df_coadapt     = load_data(coadapt_path)
        smeans_coadapt = compute_session_means(df_coadapt)
        lc_coadapt     = compute_lc_stats(smeans_coadapt)
        print(
            f"  {len(df_coadapt)} rows  |  {df_coadapt['session_id'].nunique()} sessions  |  "
            f"{df_coadapt['fold'].nunique()} folds  |  support: {sorted(df_coadapt['support_samples'].unique())}"
        )
        print("\nGenerating coadapt stats …")
        pv_coadapt = compute_pvalues(smeans_coadapt)
        print_pvalues(pv_coadapt, label="Coadapt")
        pv_coadapt.to_csv(out_dir / "pvalues_coadapt.csv", index=False)
        print(f"  P-value table → {out_dir}/pvalues_coadapt.csv")

        hs_coadapt = compute_headroom_stats(smeans_coadapt)
        print_headroom_stats(hs_coadapt, label="Coadapt")
        hs_coadapt.to_csv(out_dir / "headroom_stats_coadapt.csv", index=False)

        sens_coadapt = compute_mcnemar_sensitivity(smeans_coadapt)
        print_mcnemar_sensitivity(sens_coadapt, label="Coadapt")
        sens_coadapt.to_csv(out_dir / "mcnemar_sensitivity_coadapt.csv", index=False)

        cm = compute_crossmethod_stats(smeans, smeans_coadapt, "MAML", "Coadapt")
        print_crossmethod_stats(cm, "MAML", "Coadapt")
        cm.to_csv(out_dir / "crossmethod_maml_vs_coadapt.csv", index=False)

        if smeans_recon is not None:
            cm2 = compute_crossmethod_stats(smeans_recon, smeans_coadapt, "Recon", "Coadapt")
            print_crossmethod_stats(cm2, "Recon", "Coadapt")
            cm2.to_csv(out_dir / "crossmethod_recon_vs_coadapt.csv", index=False)

    # ── Multi-method comparison figures ───────────────────────────────────────
    extra = {}
    if smeans_recon is not None:
        extra["Recon"] = lc_recon
    if smeans_coadapt is not None:
        extra["Coadapt"] = lc_coadapt

    if extra:
        lc_dict = {"MAML": lc, **extra}
        smeans_dict = {"MAML": smeans}
        if smeans_recon is not None:
            smeans_dict["Recon"] = smeans_recon
        if smeans_coadapt is not None:
            smeans_dict["Coadapt"] = smeans_coadapt

        print("\nGenerating multi-method comparison figures …")
        fig7_tta_comparison(lc_dict, out_dir)
        fig8_all_methods(lc_dict, lc, out_dir)
        fig9_comparison_violin(smeans_dict, out_dir)


if __name__ == "__main__":
    main()
