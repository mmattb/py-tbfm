#!/usr/bin/env python3
"""
Wilcoxon Signed-Rank Test Analysis at Three Levels

1. Distribution-wide: Compare vanilla vs TTA (mean and median) across sessions
2. Per-session: Skewness of each session's TTA distribution across folds
3. Combined: Pool all TTA evaluations with their vanilla baselines
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("results/cv_analysis")

def load_data():
    """Load the analysis data."""
    session_stats = pd.read_csv(OUTPUT_DIR / "session_statistics.csv")
    full_tta = pd.read_csv(OUTPUT_DIR / "full_tta_results_with_baseline.csv")
    return session_stats, full_tta


def distribution_wide_analysis(session_stats):
    """
    Level 1: Distribution-wide Wilcoxon tests
    Compare vanilla vs TTA using session-level mean and median
    """
    print("=" * 60)
    print("LEVEL 1: DISTRIBUTION-WIDE ANALYSIS")
    print("=" * 60)
    print(f"N = {len(session_stats)} sessions\n")

    vanilla = session_stats['vanilla_test_r2'].values
    tta_mean = session_stats['tta_r2_mean'].values
    tta_median = session_stats['tta_r2_median'].values

    # Using TTA mean
    diff_mean = tta_mean - vanilla
    stat_mean, p_mean = stats.wilcoxon(diff_mean, alternative='two-sided')

    print("Using TTA MEAN per session:")
    print(f"  Vanilla mean:     {vanilla.mean():.4f} ± {vanilla.std():.4f}")
    print(f"  TTA mean:         {tta_mean.mean():.4f} ± {tta_mean.std():.4f}")
    print(f"  Mean difference:  {diff_mean.mean():.4f} ± {diff_mean.std():.4f}")
    print(f"  Wilcoxon stat:    {stat_mean}")
    print(f"  p-value:          {p_mean:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if p_mean < 0.05 else 'No'}")
    print()

    # Using TTA median
    diff_median = tta_median - vanilla
    stat_median, p_median = stats.wilcoxon(diff_median, alternative='two-sided')

    print("Using TTA MEDIAN per session:")
    print(f"  Vanilla mean:     {vanilla.mean():.4f} ± {vanilla.std():.4f}")
    print(f"  TTA median mean:  {tta_median.mean():.4f} ± {tta_median.std():.4f}")
    print(f"  Mean difference:  {diff_median.mean():.4f} ± {diff_median.std():.4f}")
    print(f"  Wilcoxon stat:    {stat_median}")
    print(f"  p-value:          {p_median:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if p_median < 0.05 else 'No'}")
    print()

    # Effect sizes (rank-biserial correlation)
    # r = 1 - (2W / n(n+1)) where W is the smaller of W+ and W-
    n = len(session_stats)
    r_mean = 1 - (2 * stat_mean) / (n * (n + 1) / 2)
    r_median = 1 - (2 * stat_median) / (n * (n + 1) / 2)

    print("Effect sizes (rank-biserial correlation r):")
    print(f"  Using TTA mean:   r = {r_mean:.4f}")
    print(f"  Using TTA median: r = {r_median:.4f}")
    print("  (|r| < 0.3 = small, 0.3-0.5 = medium, > 0.5 = large)")
    print()

    return {
        'tta_mean': {'stat': stat_mean, 'p': p_mean, 'effect_size': r_mean},
        'tta_median': {'stat': stat_median, 'p': p_median, 'effect_size': r_median}
    }


def per_session_analysis(full_tta):
    """
    Level 2: Per-session analysis
    - One-sample Wilcoxon test: TTA values vs vanilla baseline
    - Skewness of each session's TTA distribution across folds
    """
    print("=" * 60)
    print("LEVEL 2: PER-SESSION ANALYSIS")
    print("=" * 60)

    session_results = []

    for session_id in full_tta['session_id'].unique():
        session_rows = full_tta[full_tta['session_id'] == session_id]
        tta_values = session_rows['tta_r2'].values
        vanilla = session_rows['vanilla_test_r2'].iloc[0]  # Same for all rows

        if len(tta_values) >= 3:
            # One-sample Wilcoxon: test if TTA values differ from vanilla
            # Subtract vanilla and test if median differs from 0
            differences = tta_values - vanilla

            # Need non-zero differences for Wilcoxon
            nonzero_diff = differences[differences != 0]
            if len(nonzero_diff) >= 1:
                wilcox_stat, wilcox_p = stats.wilcoxon(nonzero_diff, alternative='two-sided')
            else:
                wilcox_stat, wilcox_p = np.nan, np.nan

            # Skewness and kurtosis
            skewness = stats.skew(tta_values)
            kurtosis = stats.kurtosis(tta_values)

            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(tta_values)

            # Direction: how many folds beat vanilla?
            n_better = (tta_values > vanilla).sum()
            n_worse = (tta_values < vanilla).sum()

            session_results.append({
                'session_id': session_id,
                'n_folds': len(tta_values),
                'vanilla_r2': vanilla,
                'tta_mean': tta_values.mean(),
                'tta_std': tta_values.std(),
                'mean_diff': differences.mean(),
                'wilcoxon_stat': wilcox_stat,
                'wilcoxon_p': wilcox_p,
                'significant_05': wilcox_p < 0.05 if not np.isnan(wilcox_p) else np.nan,
                'n_better': n_better,
                'n_worse': n_worse,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'normal_at_05': shapiro_p > 0.05
            })

    results_df = pd.DataFrame(session_results)

    print(f"N = {len(results_df)} sessions\n")

    # One-sample Wilcoxon results
    print("-" * 40)
    print("ONE-SAMPLE WILCOXON TEST (TTA vs Vanilla)")
    print("-" * 40)
    n_sig = results_df['significant_05'].sum()
    print(f"  Sessions significant at α=0.05: {int(n_sig)}/{len(results_df)}")

    # Of significant sessions, how many improved vs worsened?
    sig_sessions = results_df[results_df['significant_05'] == True]
    if len(sig_sessions) > 0:
        n_sig_better = (sig_sessions['mean_diff'] > 0).sum()
        n_sig_worse = (sig_sessions['mean_diff'] < 0).sum()
        print(f"    - Significantly BETTER: {n_sig_better}")
        print(f"    - Significantly WORSE:  {n_sig_worse}")
    print()

    # Sessions with strongest effects
    print("Most significant sessions (lowest p-value):")
    top_sig = results_df.nsmallest(5, 'wilcoxon_p')[['session_id', 'wilcoxon_p', 'mean_diff', 'n_folds']]
    for _, row in top_sig.iterrows():
        direction = "+" if row['mean_diff'] > 0 else ""
        print(f"  {row['session_id']}: p={row['wilcoxon_p']:.4f}, diff={direction}{row['mean_diff']:.4f} (n={row['n_folds']})")
    print()

    # Summary of direction across all sessions
    print("Direction summary (all sessions):")
    all_better = (results_df['mean_diff'] > 0).sum()
    all_worse = (results_df['mean_diff'] < 0).sum()
    print(f"  TTA mean > Vanilla: {all_better}/{len(results_df)} sessions")
    print(f"  TTA mean < Vanilla: {all_worse}/{len(results_df)} sessions")
    print()

    # Skewness summary
    print("-" * 40)
    print("SKEWNESS ANALYSIS")
    print("-" * 40)
    print(f"  Mean skewness:    {results_df['skewness'].mean():.4f}")
    print(f"  Median skewness:  {results_df['skewness'].median():.4f}")
    print(f"  Range:            [{results_df['skewness'].min():.4f}, {results_df['skewness'].max():.4f}]")
    print()

    # Categorize skewness
    n_negative = (results_df['skewness'] < -0.5).sum()
    n_symmetric = ((results_df['skewness'] >= -0.5) & (results_df['skewness'] <= 0.5)).sum()
    n_positive = (results_df['skewness'] > 0.5).sum()

    print("Skewness categories:")
    print(f"  Left-skewed (< -0.5):    {n_negative} sessions")
    print(f"  Symmetric (-0.5 to 0.5): {n_symmetric} sessions")
    print(f"  Right-skewed (> 0.5):    {n_positive} sessions")
    print()

    # Normality test results
    n_normal = results_df['normal_at_05'].sum()
    print(f"Shapiro-Wilk normality (α=0.05):")
    print(f"  Normal:     {int(n_normal)}/{len(results_df)} sessions")
    print(f"  Non-normal: {len(results_df) - int(n_normal)}/{len(results_df)} sessions")
    print()

    # Save to CSV
    results_df.to_csv(OUTPUT_DIR / "per_session_wilcoxon.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'per_session_wilcoxon.csv'}")
    print()

    return results_df


def combined_analysis(full_tta):
    """
    Level 3: Combined/pooled analysis
    Pair each TTA evaluation with its session's vanilla baseline
    """
    print("=" * 60)
    print("LEVEL 3: COMBINED (POOLED) ANALYSIS")
    print("=" * 60)

    # Each TTA observation paired with its session's vanilla baseline
    tta_values = full_tta['tta_r2'].values
    vanilla_values = full_tta['vanilla_test_r2'].values  # Already merged in full_tta

    print(f"N = {len(tta_values)} paired observations (all fold-session combinations)\n")

    differences = tta_values - vanilla_values

    print("Pooled comparison:")
    print(f"  Vanilla mean:     {vanilla_values.mean():.4f} ± {vanilla_values.std():.4f}")
    print(f"  TTA mean:         {tta_values.mean():.4f} ± {tta_values.std():.4f}")
    print(f"  Mean difference:  {differences.mean():.4f} ± {differences.std():.4f}")
    print()

    # Wilcoxon test on all pooled differences
    stat, p = stats.wilcoxon(differences, alternative='two-sided')

    print("Wilcoxon signed-rank test (pooled):")
    print(f"  Statistic:        {stat}")
    print(f"  p-value:          {p:.2e}")
    print(f"  Significant (α=0.05): {'Yes' if p < 0.05 else 'No'}")
    print()

    # Effect size
    n = len(differences)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)

    print(f"Effect size (rank-biserial r): {r:.4f}")
    print()

    # Count improvements
    n_improved = (differences > 0).sum()
    n_worsened = (differences < 0).sum()
    n_tied = (differences == 0).sum()

    print("Direction of differences:")
    print(f"  TTA > Vanilla:    {n_improved} ({n_improved/n*100:.1f}%)")
    print(f"  TTA < Vanilla:    {n_worsened} ({n_worsened/n*100:.1f}%)")
    print(f"  TTA = Vanilla:    {n_tied} ({n_tied/n*100:.1f}%)")
    print()

    # Sign test as additional check
    sign_result = stats.binomtest(n_improved, n_improved + n_worsened, p=0.5, alternative='two-sided')

    print("Sign test (binomial):")
    print(f"  p-value:          {sign_result.pvalue:.2e}")
    print(f"  Significant (α=0.05): {'Yes' if sign_result.pvalue < 0.05 else 'No'}")
    print()

    # Note about non-independence
    print("⚠️  CAUTION: These pooled observations are NOT independent!")
    print("   Sessions appear in multiple folds, inflating N artificially.")
    print("   The distribution-wide test (Level 1) is more appropriate for inference.")
    print()

    return {
        'stat': stat,
        'p': p,
        'effect_size': r,
        'n_improved': n_improved,
        'n_worsened': n_worsened
    }


def main():
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST ANALYSIS")
    print("=" * 60 + "\n")

    session_stats, full_tta = load_data()

    # Level 1: Distribution-wide
    dist_results = distribution_wide_analysis(session_stats)

    # Level 2: Per-session Wilcoxon
    session_df = per_session_analysis(full_tta)

    # Level 3: Combined/pooled
    pooled_results = combined_analysis(full_tta)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Level 1 (Distribution-wide, n=40 sessions):")
    print(f"  Using TTA mean:   p = {dist_results['tta_mean']['p']:.4f}, r = {dist_results['tta_mean']['effect_size']:.4f}")
    print(f"  Using TTA median: p = {dist_results['tta_median']['p']:.4f}, r = {dist_results['tta_median']['effect_size']:.4f}")
    print()
    print("Level 2 (Per-session one-sample Wilcoxon):")
    n_sig = int(session_df['significant_05'].sum())
    n_sig_better = int((session_df[session_df['significant_05'] == True]['mean_diff'] > 0).sum())
    n_sig_worse = n_sig - n_sig_better
    print(f"  Significant at α=0.05: {n_sig}/40 sessions")
    print(f"    - Significantly better: {n_sig_better}")
    print(f"    - Significantly worse:  {n_sig_worse}")
    print(f"  Mean skewness: {session_df['skewness'].mean():.4f}")
    print()
    print("Level 3 (Pooled, n=400 - use with caution):")
    print(f"  p = {pooled_results['p']:.2e}, r = {pooled_results['effect_size']:.4f}")
    print(f"  Improved: {pooled_results['n_improved']}/400, Worsened: {pooled_results['n_worsened']}/400")
    print()


if __name__ == "__main__":
    main()
