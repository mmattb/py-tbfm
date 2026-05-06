# TBFM Cross-Validation Analysis — Recon Sweep

## Overview

This package contains the cross-validation results and analysis comparing three test-time adaptation (TTA) strategies for the **Temporal Basis Function Model (TBFM)** in the multi-session neural decoding setting.

The model is trained on 20 held-in sessions and evaluated on 20 held-out sessions via a 20-fold random cross-validation scheme. At test time, per-session stimulus embeddings (and optionally the autoencoder) are adapted using a small support set of trials, then evaluated on the remaining query trials.

The three TTA strategies compared are:

| Label | Description |
|---|---|
| **MAML** | Meta-learning (MAML-style) TTA. Inner loop adapts stimulus embeddings on support; outer loop updates shared weights on query. AE reconstruction loss **not** included in the TTA outer loop. |
| **Recon** | Same trained models and TTA procedure as MAML, but the AE reconstruction loss (`lambda_ae_recon = 0.03`) is included in the TTA outer loop alongside the prediction MSE. |
| **Coadapt** | Co-adaptation — no inner loop; AE and embeddings are jointly optimised in a single loop. Trained from a separate set of random folds. |

MAML and Recon share identical trained model weights (all folds from `random_folds_20260101_211200`, trained with `lambda_ae_recon = 0.03`). The comparison between them is a pure ablation of whether the reconstruction loss is active at test time.

All methods are compared against a **Vanilla** baseline: a model trained on a single held-in session, evaluated on held-out sessions without any adaptation.

---

## Contents

```
tbfm_cv_data.csv              # Packaged results — MAML TTA
tbfm_cv_data_recon.csv        # Packaged results — Recon TTA
tbfm_cv_data_coadapt.csv      # Packaged results — Coadapt TTA

analysis/
  portable_analysis.py        # Main analysis + figures script

figures/
  fig1_learning_curve.png
  fig2_violin.png
  fig3_fold_variability.png
  fig4_difficulty_scatter.png
  fig5_cdf.png
  fig6_mcnemar_sensitivity.png
  fig7_tta_comparison.png
  fig8_all_methods.png
  fig9_comparison_violin.png
  pvalues_maml.csv / pvalues_recon.csv / pvalues_coadapt.csv
  headroom_stats_maml.csv / headroom_stats_recon.csv / headroom_stats_coadapt.csv
  mcnemar_sensitivity_maml.csv / mcnemar_sensitivity_recon.csv / mcnemar_sensitivity_coadapt.csv
  crossmethod_recon_vs_maml.csv
  crossmethod_maml_vs_coadapt.csv
  crossmethod_recon_vs_coadapt.csv
```

---

## Data files

Each CSV has one row per `(session, fold, support_size)` triple.

| Column | Description |
|---|---|
| `session_id` | Session identifier, e.g. `MonkeyG_20150914_Session1_S1` |
| `monkey` | Subject (`MonkeyG` or `MonkeyJ`) |
| `area` | Recording area (`S1` or `M1`) |
| `num_channels` | Number of recorded channels in that session |
| `fold` | Cross-validation fold index (0–19) |
| `support_samples` | Number of support trials used for adaptation (500, 1000, 2500, 5000) |
| `tta_r2` | R² on held-out query trials after TTA |
| `vanilla_train_r2` | R² of the vanilla single-session model on its training data |
| `vanilla_test_r2` | R² of the vanilla single-session model on held-out test data (baseline) |
| `improvement_vs_vanilla` | `tta_r2 − vanilla_test_r2` |

**Coverage:** 40 sessions × 20 folds × 4 support sizes = 1600 rows per file (1581 for Coadapt due to a small number of missing folds).

---

## Reproducing the analysis

```bash
python analysis/portable_analysis.py \
  --data tbfm_cv_data.csv \
  --recon-data tbfm_cv_data_recon.csv \
  --coadapt-data tbfm_cv_data_coadapt.csv \
  --output-dir figures/
```

Single-method analysis (e.g. Recon only vs Vanilla):
```bash
python analysis/portable_analysis.py --data tbfm_cv_data_recon.csv --output-dir figures_recon/
```

---

## Figures

### Fig 1 — Learning curve (MAML vs Vanilla)
Mean R² ± 95% CI (error bars) and 95% prediction interval (shaded) for MAML TTA and Vanilla, plotted across support sizes. The CI captures uncertainty about the mean across sessions; the PI gives the expected range for a new unseen session.

### Fig 2 — Violin: MAML vs Vanilla
Left panel: split violin of session-mean R² distributions for MAML TTA and Vanilla at each support size. Right panel: distribution of per-session improvement (TTA − Vanilla).

### Fig 3 — Cross-fold variability
For each session, each fold's R² is expressed as a deviation from that session's mean across folds. Shows how much fold assignment (i.e., which 20 sessions are held in) affects adaptation quality for a given held-out session.

### Fig 4 — Difficulty scatter (headroom-corrected)
One panel per support size. X-axis: vanilla R² (session difficulty). Y-axis: headroom-corrected improvement = `(TTA − vanilla) / (1 − vanilla)`. Points coloured by area (S1/M1). Spearman ρ and p-value annotated. Tests whether TTA disproportionately benefits harder sessions.

### Fig 5 — CDF: hard vs easy sessions
Empirical CDFs of headroom improvement split at the median vanilla R² (hard = below median, easy = above). Shaded area shows CDF separation. Annotated with probability of superiority P̂(hard > easy) and one-sided Mann-Whitney p-value.

### Fig 6 — McNemar heatmap
Rows = R² thresholds (0.05, 0.10, 0.15, 0.20); columns = support sizes. Each cell shows rescued / broken session counts and −log₁₀(McNemar p). A "rescued" session was below threshold under Vanilla but above under TTA; "broken" is the reverse. Tests whether TTA rescues more sessions than it breaks.

### Fig 7 — All TTA methods learning curve
MAML, Recon, and Coadapt mean R² ± 95% CI on a single axis, without Vanilla. Directly compares how adaptation performance scales with support size across strategies.

### Fig 8 — Vanilla + all TTA methods
Same as Fig 7 but includes the Vanilla baseline. Shows absolute performance levels for all four conditions.

### Fig 9 — Comparison violin
Violin of session-mean R² for all methods (MAML, Recon, Coadapt, Vanilla) at each support size in separate panels. Provides a full distributional view of the four-way comparison.

---

## Statistics outputs

### `pvalues_<method>.csv`
Per support size, testing each TTA method against the Vanilla baseline:

| Column | Test |
|---|---|
| `win_rate` | Fraction of sessions where TTA R² > Vanilla R² |
| `win_rate_p` | Sign test (binomial), H₁: P(win) > 50%, one-sided |
| `wilcoxon_p` | Wilcoxon signed-rank on session means, H₁: improvement > 0, one-sided |
| `BF_p` | Brown-Forsythe (Levene with median centering), H₁: var(Vanilla) > var(TTA), one-sided |
| `mcnemar_p` | McNemar on R² < 0.05 poor-fit sessions, H₁: rescued > broken |

### `headroom_stats_<method>.csv`
Per support size, testing whether harder sessions benefit more from TTA:

| Column | Test |
|---|---|
| `spearman_rho` / `spearman_p` | Spearman ρ(vanilla R², headroom improvement), H₁: ρ < 0 |
| `mw_U` / `mw_p` | Mann-Whitney U (hard vs easy sessions split at median vanilla R²), H₁: hard > easy |

### `mcnemar_sensitivity_<method>.csv`
McNemar test swept across four R² thresholds (0.05, 0.10, 0.15, 0.20) and all support sizes, with rescued / broken counts.

### `crossmethod_<A>_vs_<B>.csv`
Pairwise comparison between two TTA methods on session means (inner-joined on session ID). Wilcoxon signed-rank, H₁: A > B, one-sided.

---

## Key findings

**All three methods beat Vanilla** — sign test and Wilcoxon are significant at small support sizes for all methods. Recon maintains significance across all four support sizes (p < 10⁻⁶ at 2500 and 5000); MAML loses significance at larger support (ns at 5000), suggesting the recon loss helps maintain adaptation quality as more data becomes available.

**Recon ≈ MAML** — direct comparison shows no significant difference at any support size (all ns, win rates 45–57%). Including the AE reconstruction loss in the TTA outer loop does not significantly hurt or help adaptation quality relative to omitting it.

**Both MAML and Recon beat Coadapt at large support** — strongly significant from 1000 samples upward (p < 10⁻⁶), with 97.5% win rates at 5000 samples. At 500 samples Coadapt is more competitive (~65% win rate, p ≈ 0.015).

**Harder sessions benefit more** — consistent negative Spearman ρ (−0.5 to −0.75) for all methods: sessions that are harder for the vanilla model gain proportionally more from TTA. This holds across all support sizes and all three strategies.

**Poor-fit session rescue** — TTA rescues nearly all vanilla poor-fit sessions (R² < 0.05) while breaking essentially none. Recon rescues 18/19 poor-fit sessions consistently across all support sizes; MAML rescues fewer at higher support (3/3 at 5000).
