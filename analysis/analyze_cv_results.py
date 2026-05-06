#!/usr/bin/env python3
"""
Comprehensive Cross-Validation Results Analysis

Analyzes results from random_folds_20260101_211200 and compares
with vanilla baseline from state_dependency_results_vanilla.
Supports multiple TTA configurations with different support sample sizes.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/home/danmuir/GitHub/py-tbfm")
RANDOM_FOLDS_DIR = BASE_DIR / "random_folds_20260101_211200"

# TTA directories with different support sample sizes
TTA_CONFIGS = {
    500: RANDOM_FOLDS_DIR / "tta_results_500_20260219_032601",
    1000: RANDOM_FOLDS_DIR / "tta_results_1000_20260126_174442",
    2500: RANDOM_FOLDS_DIR / "tta_results_2500_20260202_075030",
    5000: RANDOM_FOLDS_DIR / "tta_results_5000_20260108_235311",
}

# Vanilla directories with different support sample sizes (corresponding baselines)
VANILLA_CONFIGS = {
    500: BASE_DIR / "state_dependency_results_vanilla_500",
    1000: BASE_DIR / "state_dependency_results_vanilla_1000",
    2500: BASE_DIR / "state_dependency_results_vanilla_2500",
    5000: BASE_DIR / "state_dependency_results_vanilla_5000",
}

# Shared folds will be computed dynamically as the intersection across all configs
SHARED_FOLDS = None  # Computed at runtime in parse_tta_results()
OUTPUT_DIR = BASE_DIR / "results" / "cv_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_vanilla_results(support_samples=None):
    """Parse vanilla baseline results from summary files.

    Args:
        support_samples: If specified, only load results for this support sample size.
                        If None, load all available configurations.

    Returns:
        DataFrame with vanilla results, including 'support_samples' column.
    """
    all_data = []

    configs_to_load = {support_samples: VANILLA_CONFIGS[support_samples]} if support_samples else VANILLA_CONFIGS

    for n_support, vanilla_dir in configs_to_load.items():
        if not vanilla_dir.exists():
            print(f"  Warning: Vanilla directory not found for {n_support} support: {vanilla_dir}")
            continue

        session_count = 0
        for session_dir in vanilla_dir.iterdir():
            if not session_dir.is_dir():
                continue

            summary_file = session_dir / "summary.txt"
            if not summary_file.exists():
                continue

            content = summary_file.read_text()

            # Parse values
            session_match = re.search(r"Session: (\S+)", content)
            channels_match = re.search(r"Number of channels: (\d+)", content)
            train_r2_match = re.search(r"Train R²: ([\d.-]+)", content)
            test_r2_match = re.search(r"Test R²: ([\d.-]+)", content)

            if all([session_match, channels_match, train_r2_match, test_r2_match]):
                session_id = session_match.group(1)

                # Parse session metadata
                parts = session_id.split('_')
                monkey = parts[0]
                date = parts[1]
                session_num = parts[2]
                area = parts[3] if len(parts) > 3 else 'S1'

                all_data.append({
                    'session_id': session_id,
                    'support_samples': n_support,
                    'monkey': monkey,
                    'date': date,
                    'session_num': session_num,
                    'area': area,
                    'num_channels': int(channels_match.group(1)),
                    'vanilla_train_r2': float(train_r2_match.group(1)),
                    'vanilla_test_r2': float(test_r2_match.group(1))
                })
                session_count += 1

        print(f"  Loaded {session_count} sessions for {n_support} support samples")

    return pd.DataFrame(all_data)


def parse_tta_results(support_samples=None):
    """Parse TTA results from the random folds experiment.

    Loads from tta_summary.csv first, then fills in any missing folds
    by reading per-fold CSV files from fold subdirectories.

    Args:
        support_samples: If specified, only load results for this support sample size.
                        If None, load all available configurations.

    Returns:
        DataFrame with TTA results, including 'support_samples' column.
    """
    global SHARED_FOLDS

    all_dfs = []
    folds_per_config = {}

    configs_to_load = {support_samples: TTA_CONFIGS[support_samples]} if support_samples else TTA_CONFIGS

    for n_support, tta_dir in configs_to_load.items():
        rows = []

        # First try the summary CSV
        tta_summary = tta_dir / "tta_summary.csv"
        summary_folds = set()
        if tta_summary.exists():
            df_summary = pd.read_csv(tta_summary)
            summary_folds = set(df_summary['fold'].unique())

        # Then scan fold directories for per-session CSVs to fill gaps
        for fold_dir in sorted(tta_dir.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith('fold'):
                continue
            try:
                fold_num = int(fold_dir.name.replace('fold', ''))
            except ValueError:
                continue

            # Skip if already in summary CSV
            if fold_num in summary_folds:
                continue

            # Look for per-session CSV
            per_session_csvs = list(fold_dir.glob("*_per_session.csv"))
            if not per_session_csvs:
                continue

            per_session_df = pd.read_csv(per_session_csvs[0])
            for _, row in per_session_df.iterrows():
                rows.append({
                    'fold': fold_num,
                    'session_id': row['session_id'],
                    'strategy': row['strategy'],
                    'support_size': n_support,
                    'tta_r2': row['session_r2'],
                    'start_time': '',
                    'end_time': '',
                    'duration_sec': 0,
                    'status': 'SUCCESS'
                })

        # Combine summary CSV + per-fold data
        if tta_summary.exists():
            df_summary = df_summary.rename(columns={'r2': 'tta_r2'})
            df_summary['support_samples'] = n_support
            parts = [df_summary]
        else:
            parts = []

        if rows:
            df_extra = pd.DataFrame(rows)
            df_extra['support_samples'] = n_support
            parts.append(df_extra)

        if not parts:
            print(f"  Warning: No TTA data found for {n_support} support: {tta_dir}")
            continue

        df = pd.concat(parts, ignore_index=True)
        folds_per_config[n_support] = set(df['fold'].unique())
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No TTA summary files found")

    for n_support, folds in folds_per_config.items():
        print(f"  {n_support} support: {len(folds)} folds, folds {sorted(folds)}")

    SHARED_FOLDS = sorted(set().union(*folds_per_config.values()))

    combined = pd.concat(all_dfs, ignore_index=True)
    for n_support in sorted(combined['support_samples'].unique()):
        n_eval = len(combined[combined['support_samples'] == n_support])
        print(f"  {n_support} support: {n_eval} total evaluations")

    return combined


def compute_session_statistics(vanilla_df, tta_df):
    """Compute comprehensive per-session statistics.

    Args:
        vanilla_df: DataFrame with vanilla baseline results (includes support_samples column).
        tta_df: DataFrame with TTA results (includes support_samples column).

    Returns:
        DataFrame with session statistics for each support sample configuration.
        For support sizes without vanilla baselines, vanilla columns will be NaN.
    """
    support_sizes = sorted(tta_df['support_samples'].unique())

    stats_list = []

    for n_support in support_sizes:
        tta_subset = tta_df[tta_df['support_samples'] == n_support]
        vanilla_subset = vanilla_df[vanilla_df['support_samples'] == n_support]

        has_vanilla = not vanilla_subset.empty

        if not has_vanilla:
            print(f"  Note: No vanilla results for {n_support} support samples (TTA-only stats)")

        # Use all sessions from TTA data (not just those in vanilla)
        sessions = tta_subset['session_id'].unique()

        for session_id in sessions:
            tta_rows = tta_subset[tta_subset['session_id'] == session_id]
            tta_r2_values = tta_rows['tta_r2'].values

            # Parse session metadata from session_id
            parts = session_id.split('_')
            monkey = parts[0]
            date = parts[1]
            session_num = parts[2]
            area = parts[3] if len(parts) > 3 else 'S1'

            # Get vanilla data if available
            vanilla_row = None
            if has_vanilla:
                v_match = vanilla_subset[vanilla_subset['session_id'] == session_id]
                if len(v_match) > 0:
                    vanilla_row = v_match.iloc[0]

            vanilla_train = vanilla_row['vanilla_train_r2'] if vanilla_row is not None else np.nan
            vanilla_test = vanilla_row['vanilla_test_r2'] if vanilla_row is not None else np.nan
            num_channels = vanilla_row['num_channels'] if vanilla_row is not None else np.nan

            tta_mean = np.mean(tta_r2_values) if len(tta_r2_values) > 0 else np.nan

            stats = {
                'session_id': session_id,
                'support_samples': n_support,
                'monkey': monkey,
                'date': date,
                'session_num': session_num,
                'area': area,
                'num_channels': num_channels,

                # Vanilla baseline
                'vanilla_train_r2': vanilla_train,
                'vanilla_test_r2': vanilla_test,

                # TTA statistics
                'tta_r2_mean': tta_mean,
                'tta_r2_std': np.std(tta_r2_values) if len(tta_r2_values) > 0 else np.nan,
                'tta_r2_min': np.min(tta_r2_values) if len(tta_r2_values) > 0 else np.nan,
                'tta_r2_max': np.max(tta_r2_values) if len(tta_r2_values) > 0 else np.nan,
                'tta_r2_median': np.median(tta_r2_values) if len(tta_r2_values) > 0 else np.nan,
                'tta_r2_q25': np.percentile(tta_r2_values, 25) if len(tta_r2_values) > 0 else np.nan,
                'tta_r2_q75': np.percentile(tta_r2_values, 75) if len(tta_r2_values) > 0 else np.nan,
                'tta_num_folds': len(tta_r2_values),

                # Comparisons (NaN if no vanilla)
                'improvement_vs_vanilla_test': (tta_mean - vanilla_test) if (not np.isnan(vanilla_test) and len(tta_r2_values) > 0) else np.nan,
                'relative_improvement': ((tta_mean - vanilla_test) / max(abs(vanilla_test), 0.01)) * 100 if (not np.isnan(vanilla_test) and len(tta_r2_values) > 0) else np.nan,
            }

            stats_list.append(stats)

    return pd.DataFrame(stats_list)


def compute_fold_statistics(tta_df):
    """Compute per-fold statistics for each support sample configuration."""
    fold_stats = tta_df.groupby(['support_samples', 'fold']).agg({
        'tta_r2': ['mean', 'std', 'min', 'max', 'median'],
        'duration_sec': 'first'
    }).reset_index()

    fold_stats.columns = ['support_samples', 'fold', 'mean_r2', 'std_r2', 'min_r2', 'max_r2', 'median_r2', 'duration_sec']
    fold_stats['duration_hours'] = fold_stats['duration_sec'] / 3600

    return fold_stats


def create_comprehensive_csv(session_stats, tta_df, fold_stats, vanilla_df):
    """Create comprehensive CSV with all important data points."""

    # Main session-level analysis
    session_stats.to_csv(OUTPUT_DIR / "session_statistics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'session_statistics.csv'}")

    # Fold-level statistics
    fold_stats.to_csv(OUTPUT_DIR / "fold_statistics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'fold_statistics.csv'}")

    # Full TTA results with vanilla reference (matching by support_samples)
    full_tta = tta_df.merge(
        vanilla_df[['session_id', 'support_samples', 'monkey', 'area', 'num_channels', 'vanilla_train_r2', 'vanilla_test_r2']],
        on=['session_id', 'support_samples'],
        how='left'
    )
    full_tta['improvement_vs_vanilla'] = full_tta['tta_r2'] - full_tta['vanilla_test_r2']
    full_tta.to_csv(OUTPUT_DIR / "full_tta_results_with_baseline.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'full_tta_results_with_baseline.csv'}")

    # Summary statistics
    summary_rows = []

    support_sizes = sorted(tta_df['support_samples'].unique())

    for n_support in support_sizes:
        tta_subset = tta_df[tta_df['support_samples'] == n_support]
        vanilla_subset = vanilla_df[vanilla_df['support_samples'] == n_support]
        session_subset = session_stats[session_stats['support_samples'] == n_support]

        # Overall TTA performance
        summary_rows.append({'support_samples': n_support, 'metric': 'TTA R² Mean', 'value': tta_subset['tta_r2'].mean()})
        summary_rows.append({'support_samples': n_support, 'metric': 'TTA R² Std', 'value': tta_subset['tta_r2'].std()})
        summary_rows.append({'support_samples': n_support, 'metric': 'TTA R² Min', 'value': tta_subset['tta_r2'].min()})
        summary_rows.append({'support_samples': n_support, 'metric': 'TTA R² Max', 'value': tta_subset['tta_r2'].max()})

        # Vanilla baseline (may not exist for all support sizes)
        if not vanilla_subset.empty:
            summary_rows.append({'support_samples': n_support, 'metric': 'Vanilla Test R² Mean', 'value': vanilla_subset['vanilla_test_r2'].mean()})
            summary_rows.append({'support_samples': n_support, 'metric': 'Vanilla Test R² Std', 'value': vanilla_subset['vanilla_test_r2'].std()})

        # Improvement (only where vanilla exists)
        imp_col = session_subset['improvement_vs_vanilla_test'].dropna()
        avg_improvement = imp_col.mean() if len(imp_col) > 0 else np.nan
        summary_rows.append({'support_samples': n_support, 'metric': 'Avg Improvement vs Vanilla', 'value': avg_improvement})

        # Count sessions with improvement
        n_improved = (imp_col > 0).sum() if len(imp_col) > 0 else 0
        n_sessions = len(session_subset)
        summary_rows.append({'support_samples': n_support, 'metric': 'Sessions Improved (count)', 'value': n_improved})
        n_with_vanilla = len(imp_col)
        summary_rows.append({'support_samples': n_support, 'metric': 'Sessions Improved (%)', 'value': n_improved / n_with_vanilla * 100 if n_with_vanilla > 0 else np.nan})

        # Per-monkey stats
        for monkey in ['MonkeyG', 'MonkeyJ']:
            monkey_sessions = session_subset[session_subset['monkey'] == monkey]
            if len(monkey_sessions) > 0:
                summary_rows.append({'support_samples': n_support, 'metric': f'{monkey} TTA R² Mean', 'value': monkey_sessions['tta_r2_mean'].mean()})
                summary_rows.append({'support_samples': n_support, 'metric': f'{monkey} Vanilla Test R² Mean', 'value': monkey_sessions['vanilla_test_r2'].mean()})
                summary_rows.append({'support_samples': n_support, 'metric': f'{monkey} Improvement Mean', 'value': monkey_sessions['improvement_vs_vanilla_test'].mean()})

        # Per-area stats
        for area in ['S1', 'M1']:
            area_sessions = session_subset[session_subset['area'] == area]
            if len(area_sessions) > 0:
                summary_rows.append({'support_samples': n_support, 'metric': f'{area} TTA R² Mean', 'value': area_sessions['tta_r2_mean'].mean()})
                summary_rows.append({'support_samples': n_support, 'metric': f'{area} Vanilla Test R² Mean', 'value': area_sessions['vanilla_test_r2'].mean()})
                summary_rows.append({'support_samples': n_support, 'metric': f'{area} Improvement Mean', 'value': area_sessions['improvement_vs_vanilla_test'].mean()})

        # Experiment stats for this support sample size
        fold_subset = fold_stats[fold_stats['support_samples'] == n_support]
        summary_rows.append({'support_samples': n_support, 'metric': 'Total Folds', 'value': len(fold_subset)})
        summary_rows.append({'support_samples': n_support, 'metric': 'Total Sessions', 'value': n_sessions})
        summary_rows.append({'support_samples': n_support, 'metric': 'Total TTA Evaluations', 'value': len(tta_subset)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'summary_statistics.csv'}")

    return full_tta


def create_visualizations(session_stats, tta_df, fold_stats, vanilla_df):
    """Generate comprehensive visualizations."""

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    support_sizes = sorted(session_stats['support_samples'].unique())

    # 0. Support samples comparison plot (NEW - key comparison)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Performance comparison by support samples
    ax = axes[0]
    support_means = session_stats.groupby('support_samples').agg({
        'tta_r2_mean': 'mean',
        'vanilla_test_r2': 'mean'
    }).reset_index()
    x = np.arange(len(support_means))
    width = 0.35
    ax.bar(x - width/2, support_means['vanilla_test_r2'], width, label='Vanilla Test R²', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, support_means['tta_r2_mean'], width, label='TTA R² Mean', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in support_means['support_samples']])
    ax.set_xlabel('Support Samples', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Performance by Support Sample Size', fontsize=14)
    ax.legend()

    # Improvement comparison
    ax = axes[1]
    improvement_by_support = session_stats.groupby('support_samples')['improvement_vs_vanilla_test'].mean()
    colors = ['green' if x > 0 else 'red' for x in improvement_by_support.values]
    ax.bar(improvement_by_support.index.astype(str), improvement_by_support.values, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Support Samples', fontsize=12)
    ax.set_ylabel('Mean Improvement vs Vanilla', fontsize=12)
    ax.set_title('TTA Improvement by Support Size', fontsize=14)

    # Sessions improved percentage
    ax = axes[2]
    pct_improved = session_stats.groupby('support_samples').apply(
        lambda x: (x['improvement_vs_vanilla_test'] > 0).mean() * 100,
        include_groups=False
    )
    ax.bar(pct_improved.index.astype(str), pct_improved.values, color='#3498db', alpha=0.7)
    ax.set_xlabel('Support Samples', fontsize=12)
    ax.set_ylabel('Sessions Improved (%)', fontsize=12)
    ax.set_title('% Sessions with Positive Improvement', fontsize=14)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(fig_dir / "support_samples_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'support_samples_comparison.png'}")

    # 1. TTA vs Vanilla comparison scatter plot (one per support sample size)
    n_support_configs = len(support_sizes)
    fig, axes = plt.subplots(1, n_support_configs, figsize=(6 * n_support_configs, 6))
    if n_support_configs == 1:
        axes = [axes]

    colors = {'MonkeyG': '#1f77b4', 'MonkeyJ': '#ff7f0e'}
    markers = {'S1': 'o', 'M1': 's'}

    for idx, n_support in enumerate(support_sizes):
        ax = axes[idx]
        subset_stats = session_stats[session_stats['support_samples'] == n_support]

        # Filter out rows with NaN TTA values
        subset_stats = subset_stats.dropna(subset=['tta_r2_mean', 'vanilla_test_r2'])

        if len(subset_stats) == 0:
            ax.text(0.5, 0.5, f'No data for {n_support} support',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{n_support} Support Samples', fontsize=14)
            continue

        for monkey in ['MonkeyG', 'MonkeyJ']:
            for area in ['S1', 'M1']:
                subset = subset_stats[(subset_stats['monkey'] == monkey) & (subset_stats['area'] == area)]
                if len(subset) > 0:
                    ax.scatter(subset['vanilla_test_r2'], subset['tta_r2_mean'],
                              c=colors[monkey], marker=markers[area],
                              label=f'{monkey} {area}', s=100, alpha=0.7)
                    ax.errorbar(subset['vanilla_test_r2'], subset['tta_r2_mean'],
                               yerr=subset['tta_r2_std'], fmt='none',
                               c=colors[monkey], alpha=0.3, capsize=3)

        # Diagonal line (y=x)
        all_vals = np.concatenate([subset_stats['vanilla_test_r2'].values, subset_stats['tta_r2_mean'].values])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 0:
            lims = [all_vals.min() - 0.05, all_vals.max() + 0.05]
            ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        ax.set_xlabel('Vanilla Test R²', fontsize=12)
        ax.set_ylabel('TTA R² (mean ± std)', fontsize=12)
        ax.set_title(f'{n_support} Support Samples', fontsize=14)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "tta_vs_vanilla_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'tta_vs_vanilla_scatter.png'}")

    # 2. Improvement distribution (one row per support sample size)
    fig, axes = plt.subplots(n_support_configs, 2, figsize=(14, 5 * n_support_configs))
    if n_support_configs == 1:
        axes = [axes]

    for idx, n_support in enumerate(support_sizes):
        subset_stats = session_stats[session_stats['support_samples'] == n_support].dropna(subset=['improvement_vs_vanilla_test'])

        ax = axes[idx][0]
        if len(subset_stats) > 0:
            subset_stats['improvement_vs_vanilla_test'].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
            mean_imp = subset_stats['improvement_vs_vanilla_test'].mean()
            ax.axvline(mean_imp, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_imp:.3f}')
        ax.set_xlabel('Improvement (TTA R² - Vanilla Test R²)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{n_support} Support: Improvement Distribution', fontsize=14)
        ax.legend()

        ax = axes[idx][1]
        if len(subset_stats) > 0:
            improvement_by_session = subset_stats.sort_values('improvement_vs_vanilla_test')
            bar_colors = ['green' if x > 0 else 'red' for x in improvement_by_session['improvement_vs_vanilla_test']]
            ax.barh(range(len(improvement_by_session)), improvement_by_session['improvement_vs_vanilla_test'], color=bar_colors, alpha=0.7)
            ax.set_yticks(range(len(improvement_by_session)))
            ax.set_yticklabels(improvement_by_session['session_id'], fontsize=6)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Improvement (TTA R² - Vanilla Test R²)', fontsize=12)
        ax.set_title(f'{n_support} Support: Per-Session Improvement', fontsize=14)

    plt.tight_layout()
    plt.savefig(fig_dir / "improvement_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'improvement_distribution.png'}")

    # 3. Per-fold performance boxplot (one per support sample size)
    fig, axes = plt.subplots(n_support_configs, 1, figsize=(14, 6 * n_support_configs))
    if n_support_configs == 1:
        axes = [axes]

    support_colors = {500: '#9b59b6', 1000: '#e74c3c', 2500: '#f39c12', 5000: '#3498db'}

    for idx, n_support in enumerate(support_sizes):
        ax = axes[idx]
        tta_subset = tta_df[tta_df['support_samples'] == n_support]

        n_folds = tta_subset['fold'].nunique()
        fold_data = [tta_subset[tta_subset['fold'] == f]['tta_r2'].values for f in range(n_folds)]
        bp = ax.boxplot(fold_data, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor(support_colors.get(n_support, '#3498db'))
            patch.set_alpha(0.7)

        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('TTA R²', fontsize=12)
        ax.set_title(f'{n_support} Support Samples: TTA R² Distribution per Fold', fontsize=14)
        ax.set_xticks(range(1, n_folds + 1))
        ax.set_xticklabels(range(n_folds))

        # Add mean line
        fold_means = [np.mean(d) if len(d) > 0 else np.nan for d in fold_data]
        valid_means = [(i+1, m) for i, m in enumerate(fold_means) if not np.isnan(m)]
        if valid_means:
            xs, ys = zip(*valid_means)
            ax.plot(xs, ys, 'ko-', markersize=5, label='Fold Mean')
            overall_mean = np.nanmean(fold_means)
            ax.axhline(overall_mean, color='green', linestyle='--', label=f'Overall Mean: {overall_mean:.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "fold_performance_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'fold_performance_boxplot.png'}")

    # 4. Heatmap of session performance across folds (one per support sample size)
    for n_support in support_sizes:
        tta_subset = tta_df[tta_df['support_samples'] == n_support]
        pivot_df = tta_subset.pivot_table(index='session_id', columns='fold', values='tta_r2', aggfunc='first')

        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(pivot_df, cmap='RdYlGn', center=0.4, annot=False,
                    xticklabels=True, yticklabels=True, ax=ax)
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('Session', fontsize=10)
        ax.set_title(f'TTA R² Heatmap: Sessions × Folds ({n_support} Support)', fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / f"session_fold_heatmap_{n_support}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_dir / f'session_fold_heatmap_{n_support}.png'}")

    # 5. Performance by monkey and area (grouped by support samples)
    fig, axes = plt.subplots(2, n_support_configs, figsize=(5 * n_support_configs, 10))
    if n_support_configs == 1:
        axes = axes.reshape(-1, 1)

    for idx, n_support in enumerate(support_sizes):
        subset_stats = session_stats[session_stats['support_samples'] == n_support]

        # By monkey
        ax = axes[0, idx]
        monkey_data = subset_stats.groupby('monkey').agg({
            'vanilla_test_r2': 'mean',
            'tta_r2_mean': 'mean'
        }).reset_index()

        x = np.arange(len(monkey_data))
        width = 0.35
        ax.bar(x - width/2, monkey_data['vanilla_test_r2'], width, label='Vanilla Test R²', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, monkey_data['tta_r2_mean'], width, label='TTA R² Mean', color='#2ecc71', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(monkey_data['monkey'])
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title(f'{n_support} Support: By Monkey', fontsize=14)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 0.7)

        # By area
        ax = axes[1, idx]
        area_data = subset_stats.groupby('area').agg({
            'vanilla_test_r2': 'mean',
            'tta_r2_mean': 'mean'
        }).reset_index()

        x = np.arange(len(area_data))
        ax.bar(x - width/2, area_data['vanilla_test_r2'], width, label='Vanilla Test R²', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, area_data['tta_r2_mean'], width, label='TTA R² Mean', color='#2ecc71', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(area_data['area'])
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title(f'{n_support} Support: By Brain Area', fontsize=14)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 0.7)

    plt.tight_layout()
    plt.savefig(fig_dir / "performance_by_group.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'performance_by_group.png'}")

    # 6. Correlation with number of channels (colored by support samples)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    support_colors = {500: '#9b59b6', 1000: '#e74c3c', 2500: '#f39c12', 5000: '#3498db'}

    ax = axes[0]
    for n_support in support_sizes:
        subset = session_stats[session_stats['support_samples'] == n_support]
        ax.scatter(subset['num_channels'], subset['tta_r2_mean'],
                  alpha=0.6, s=60, c=support_colors.get(n_support, '#333'),
                  label=f'{n_support} support')

    # Overall fit line
    ax.set_xlabel('Number of Channels', fontsize=12)
    ax.set_ylabel('TTA R² Mean', fontsize=12)
    ax.set_title('TTA R² vs Channel Count', fontsize=14)
    ax.legend()

    ax = axes[1]
    for n_support in support_sizes:
        subset = session_stats[session_stats['support_samples'] == n_support]
        ax.scatter(subset['num_channels'], subset['improvement_vs_vanilla_test'],
                  alpha=0.6, s=60, c=support_colors.get(n_support, '#333'),
                  label=f'{n_support} support')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Channels', fontsize=12)
    ax.set_ylabel('Improvement vs Vanilla', fontsize=12)
    ax.set_title('Improvement vs Channel Count', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "channel_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'channel_correlation.png'}")

    # 7. Session variability analysis (one per support sample size)
    for n_support in support_sizes:
        subset_stats = session_stats[session_stats['support_samples'] == n_support]
        fig, ax = plt.subplots(figsize=(12, max(6, len(subset_stats) * 0.2)))

        session_stats_sorted = subset_stats.sort_values('tta_r2_std', ascending=False)

        ax.barh(range(len(session_stats_sorted)), session_stats_sorted['tta_r2_std'],
               alpha=0.7, color=support_colors.get(n_support, '#3498db'))
        ax.set_yticks(range(len(session_stats_sorted)))
        ax.set_yticklabels(session_stats_sorted['session_id'], fontsize=7)
        ax.set_xlabel('TTA R² Std (across folds)', fontsize=12)
        ax.set_title(f'{n_support} Support: Session Variability Across Folds', fontsize=14)
        ax.axvline(session_stats_sorted['tta_r2_std'].mean(), color='red', linestyle='--',
                   label=f'Mean: {session_stats_sorted["tta_r2_std"].mean():.3f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(fig_dir / f"session_variability_{n_support}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_dir / f'session_variability_{n_support}.png'}")

    # 8. Summary dashboard (with support sample comparison)
    fig = plt.figure(figsize=(18, 14))

    # Main title
    support_str = ', '.join([str(s) for s in support_sizes])
    fig.suptitle(f'Cross-Validation Analysis Summary\nSupport Samples: {support_str}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Stats text box (summary across all configs)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')

    stats_lines = ["STATISTICS BY SUPPORT SAMPLES", "─" * 30]
    for n_support in support_sizes:
        tta_sub = tta_df[tta_df['support_samples'] == n_support]
        vanilla_sub = vanilla_df[vanilla_df['support_samples'] == n_support]
        session_sub = session_stats[session_stats['support_samples'] == n_support]

        imp_col = session_sub['improvement_vs_vanilla_test'].dropna()
        n_improved = (imp_col > 0).sum()

        stats_lines.append(f"\n{n_support} SUPPORT:")
        stats_lines.append(f"  TTA R²: {tta_sub['tta_r2'].mean():.4f} ± {tta_sub['tta_r2'].std():.4f}")
        if not vanilla_sub.empty:
            stats_lines.append(f"  Vanilla: {vanilla_sub['vanilla_test_r2'].mean():.4f}")
            stats_lines.append(f"  Improvement: {imp_col.mean():.4f}" if len(imp_col) > 0 else "  Improvement: N/A")
            stats_lines.append(f"  Improved: {n_improved}/{len(imp_col)}" if len(imp_col) > 0 else "  Improved: N/A")
        else:
            stats_lines.append(f"  Vanilla: N/A")
            stats_lines.append(f"  Sessions: {len(session_sub)}")

    stats_text = "\n".join(stats_lines)
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # TTA vs Vanilla scatter (mini, colored by support samples)
    ax2 = fig.add_subplot(2, 3, 2)
    for n_support in support_sizes:
        subset = session_stats[session_stats['support_samples'] == n_support]
        ax2.scatter(subset['vanilla_test_r2'], subset['tta_r2_mean'],
                   alpha=0.6, s=30, c=support_colors.get(n_support, '#333'),
                   label=f'{n_support}')
    all_vals = np.concatenate([session_stats['vanilla_test_r2'].values, session_stats['tta_r2_mean'].values])
    lims = [all_vals.min() - 0.05, all_vals.max() + 0.05]
    ax2.plot(lims, lims, 'k--', alpha=0.5)
    ax2.set_xlabel('Vanilla Test R²')
    ax2.set_ylabel('TTA R² Mean')
    ax2.set_title('TTA vs Vanilla')
    ax2.legend(title='Support', fontsize=8)

    # Improvement histogram (stacked by support)
    ax3 = fig.add_subplot(2, 3, 3)
    for n_support in support_sizes:
        subset = session_stats[session_stats['support_samples'] == n_support]
        imp_vals = subset['improvement_vs_vanilla_test'].dropna()
        if len(imp_vals) > 0:
            ax3.hist(imp_vals, bins=15, alpha=0.5,
                    label=f'{n_support}', color=support_colors.get(n_support, '#333'))
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_xlabel('Improvement')
    ax3.set_ylabel('Count')
    ax3.set_title('Improvement Distribution')
    ax3.legend(title='Support', fontsize=8)

    # Performance comparison by support samples
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(len(support_sizes))
    width = 0.35
    tta_means = [session_stats[session_stats['support_samples'] == sz]['tta_r2_mean'].mean() for sz in support_sizes]
    vanilla_means = []
    for sz in support_sizes:
        subset = session_stats[session_stats['support_samples'] == sz]['vanilla_test_r2'].dropna()
        vanilla_means.append(subset.mean() if len(subset) > 0 else np.nan)
    # Only plot vanilla bars for sizes with data
    vanilla_arr = np.array(vanilla_means)
    has_vanilla = ~np.isnan(vanilla_arr)
    if has_vanilla.any():
        ax4.bar(x[has_vanilla] - width/2, vanilla_arr[has_vanilla], width, label='Vanilla', color='#e74c3c', alpha=0.7)
    ax4.bar(x + width/2, tta_means, width, label='TTA', color='#2ecc71', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{s}' for s in support_sizes])
    ax4.set_xlabel('Support Samples')
    ax4.set_ylabel('R²')
    ax4.set_title('Performance by Support Size')
    ax4.legend()

    # Improvement by support samples (only sizes with vanilla)
    ax5 = fig.add_subplot(2, 3, 5)
    improvement_by_support = session_stats.groupby('support_samples')['improvement_vs_vanilla_test'].mean().dropna()
    colors_imp = [support_colors.get(s, '#333') for s in improvement_by_support.index]
    ax5.bar(improvement_by_support.index.astype(str), improvement_by_support.values, color=colors_imp, alpha=0.7)
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Support Samples')
    ax5.set_ylabel('Mean Improvement')
    ax5.set_title('Improvement by Support Size')

    # Top performers across all configs
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    rank_lines = []
    for n_support in support_sizes:
        subset = session_stats[session_stats['support_samples'] == n_support]
        top3 = subset.nlargest(3, 'tta_r2_mean')[['session_id', 'tta_r2_mean']].values
        rank_lines.append(f"TOP 3 ({n_support} support):")
        for i, (s, r) in enumerate(top3, 1):
            rank_lines.append(f"  {i}. {s[:25]}: {r:.4f}")
        rank_lines.append("")

    rank_text = "\n".join(rank_lines)
    ax6.text(0.05, 0.95, rank_text, transform=ax6.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'summary_dashboard.png'}")


def wilcoxon_analysis(session_stats, tta_df, vanilla_df):
    """Perform Wilcoxon signed-rank tests comparing support sample configurations."""

    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK ANALYSIS")
    print("=" * 60)

    support_sizes = sorted(session_stats['support_samples'].unique())
    wilcoxon_results = []

    # 1. TTA vs Vanilla for each support sample size
    print("\n" + "-" * 40)
    print("TTA vs VANILLA (per support sample size)")
    print("-" * 40)

    sizes_with_vanilla = [sz for sz in support_sizes if sz in VANILLA_CONFIGS and VANILLA_CONFIGS[sz].exists()]
    for n_support in sizes_with_vanilla:
        subset = session_stats[session_stats['support_samples'] == n_support].dropna(subset=['tta_r2_mean', 'vanilla_test_r2'])

        vanilla = subset['vanilla_test_r2'].values
        tta_mean = subset['tta_r2_mean'].values
        diff = tta_mean - vanilla

        stat, p = stats.wilcoxon(diff, alternative='two-sided')
        n = len(diff)
        r = 1 - (2 * stat) / (n * (n + 1) / 2)  # rank-biserial correlation

        n_improved = (diff > 0).sum()

        print(f"\n{n_support} Support Samples (n={n}):")
        print(f"  Vanilla mean:     {vanilla.mean():.4f} ± {vanilla.std():.4f}")
        print(f"  TTA mean:         {tta_mean.mean():.4f} ± {tta_mean.std():.4f}")
        print(f"  Mean improvement: {diff.mean():.4f} ± {diff.std():.4f}")
        print(f"  Wilcoxon W:       {stat:.1f}")
        print(f"  p-value:          {p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
        print(f"  Effect size (r):  {r:.4f} ({'large' if abs(r) > 0.5 else 'medium' if abs(r) > 0.3 else 'small'})")
        print(f"  Sessions improved: {n_improved}/{n} ({n_improved/n*100:.1f}%)")

        wilcoxon_results.append({
            'comparison': f'{n_support}_TTA_vs_Vanilla',
            'n': n,
            'mean_diff': diff.mean(),
            'std_diff': diff.std(),
            'wilcoxon_stat': stat,
            'p_value': p,
            'effect_size_r': r,
            'n_improved': n_improved,
            'pct_improved': n_improved/n*100
        })

    # 2. Pairwise TTA comparisons across all support sizes
    from itertools import combinations

    stats_by_size = {}
    for sz in support_sizes:
        stats_by_size[sz] = session_stats[session_stats['support_samples'] == sz].set_index('session_id')

    print("\n" + "-" * 40)
    print("PAIRWISE TTA COMPARISONS (paired by session)")
    print("-" * 40)

    for sz_a, sz_b in combinations(support_sizes, 2):
        label_a = f'{sz_a/1000:.1f}K' if sz_a >= 1000 else str(sz_a)
        label_b = f'{sz_b/1000:.1f}K' if sz_b >= 1000 else str(sz_b)

        shared_sessions = stats_by_size[sz_a].index.intersection(stats_by_size[sz_b].index)
        n = len(shared_sessions)
        if n < 5:
            print(f"\n  Skipping {label_a} vs {label_b}: only {n} shared sessions")
            continue

        tta_a = stats_by_size[sz_a].loc[shared_sessions, 'tta_r2_mean'].values
        tta_b = stats_by_size[sz_b].loc[shared_sessions, 'tta_r2_mean'].values
        diff_tta = tta_b - tta_a  # positive means larger support is better

        stat, p = stats.wilcoxon(diff_tta, alternative='two-sided')
        r = 1 - (2 * stat) / (n * (n + 1) / 2)
        n_b_better = (diff_tta > 0).sum()

        print(f"\n{label_b} vs {label_a} TTA (n={n}):")
        print(f"  {label_a} TTA mean:  {tta_a.mean():.4f} ± {tta_a.std():.4f}")
        print(f"  {label_b} TTA mean:  {tta_b.mean():.4f} ± {tta_b.std():.4f}")
        print(f"  Diff ({label_b}-{label_a}): {diff_tta.mean():.4f} ± {diff_tta.std():.4f}")
        print(f"  Wilcoxon W:       {stat:.1f}")
        print(f"  p-value:          {p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
        print(f"  Effect size (r):  {r:.4f} ({'large' if abs(r) > 0.5 else 'medium' if abs(r) > 0.3 else 'small'})")
        print(f"  Sessions {label_b} > {label_a}: {n_b_better}/{n} ({n_b_better/n*100:.1f}%)")

        wilcoxon_results.append({
            'comparison': f'{label_b}_TTA_vs_{label_a}_TTA',
            'n': n,
            'mean_diff': diff_tta.mean(),
            'std_diff': diff_tta.std(),
            'wilcoxon_stat': stat,
            'p_value': p,
            'effect_size_r': r,
            'n_improved': n_b_better,
            'pct_improved': n_b_better/n*100
        })

    # 3. Compare improvement magnitudes where vanilla data exists for both
    print("\n" + "-" * 40)
    print("IMPROVEMENT MAGNITUDE COMPARISONS")
    print("-" * 40)

    sizes_with_vanilla = [sz for sz in support_sizes if sz in VANILLA_CONFIGS and VANILLA_CONFIGS[sz].exists()]
    for sz_a, sz_b in combinations(sizes_with_vanilla, 2):
        label_a = f'{sz_a/1000:.1f}K' if sz_a >= 1000 else str(sz_a)
        label_b = f'{sz_b/1000:.1f}K' if sz_b >= 1000 else str(sz_b)

        shared_sessions = stats_by_size[sz_a].index.intersection(stats_by_size[sz_b].index)
        # Filter to sessions that have improvement data for both
        valid_a = stats_by_size[sz_a].loc[shared_sessions].dropna(subset=['improvement_vs_vanilla_test'])
        valid_b = stats_by_size[sz_b].loc[shared_sessions].dropna(subset=['improvement_vs_vanilla_test'])
        valid_sessions = valid_a.index.intersection(valid_b.index)
        n = len(valid_sessions)
        if n < 5:
            print(f"\n  Skipping {label_a} vs {label_b} improvement: only {n} sessions with vanilla data")
            continue

        imp_a = stats_by_size[sz_a].loc[valid_sessions, 'improvement_vs_vanilla_test'].values
        imp_b = stats_by_size[sz_b].loc[valid_sessions, 'improvement_vs_vanilla_test'].values
        diff_imp = imp_a - imp_b  # positive means smaller support has larger improvement

        stat, p = stats.wilcoxon(diff_imp, alternative='two-sided')
        r = 1 - (2 * stat) / (n * (n + 1) / 2)
        n_a_bigger_imp = (diff_imp > 0).sum()

        print(f"\n{label_a} vs {label_b} improvement magnitude (n={n}):")
        print(f"  {label_a} improvement: {imp_a.mean():.4f} ± {imp_a.std():.4f}")
        print(f"  {label_b} improvement: {imp_b.mean():.4f} ± {imp_b.std():.4f}")
        print(f"  Diff ({label_a}-{label_b}): {diff_imp.mean():.4f} ± {diff_imp.std():.4f}")
        print(f"  Wilcoxon W:       {stat:.1f}")
        print(f"  p-value:          {p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
        print(f"  Effect size (r):  {r:.4f} ({'large' if abs(r) > 0.5 else 'medium' if abs(r) > 0.3 else 'small'})")
        print(f"  Sessions {label_a} > {label_b} improvement: {n_a_bigger_imp}/{n} ({n_a_bigger_imp/n*100:.1f}%)")

        wilcoxon_results.append({
            'comparison': f'{label_a}_improvement_vs_{label_b}_improvement',
            'n': n,
            'mean_diff': diff_imp.mean(),
            'std_diff': diff_imp.std(),
            'wilcoxon_stat': stat,
            'p_value': p,
            'effect_size_r': r,
            'n_improved': n_a_bigger_imp,
            'pct_improved': n_a_bigger_imp/n*100
        })

    # 4. Diminishing returns analysis: marginal gain 500->1K->2.5K->5K
    diminishing_sizes = [sz for sz in [500, 1000, 2500, 5000] if sz in stats_by_size]
    if len(diminishing_sizes) >= 2:
        print("\n" + "-" * 40)
        print("DIMINISHING RETURNS ANALYSIS")
        print("-" * 40)

        common = stats_by_size[diminishing_sizes[0]].index
        for sz in diminishing_sizes[1:]:
            common = common.intersection(stats_by_size[sz].index)
        n = len(common)

        tta_by_size = {sz: stats_by_size[sz].loc[common, 'tta_r2_mean'].values for sz in diminishing_sizes}

        gains = {}
        size_labels = {500: '500', 1000: '1K', 2500: '2.5K', 5000: '5K'}
        for i in range(len(diminishing_sizes) - 1):
            sz_a, sz_b = diminishing_sizes[i], diminishing_sizes[i + 1]
            gains[(sz_a, sz_b)] = tta_by_size[sz_b] - tta_by_size[sz_a]

        print(f"\nMarginal gain analysis (n={n} sessions):")
        for (sz_a, sz_b), gain in gains.items():
            print(f"  {size_labels[sz_a]} → {size_labels[sz_b]} gain:  {gain.mean():.4f} ± {gain.std():.4f}")

        gain_list = list(gains.values())
        gain_keys = list(gains.keys())
        if len(gain_list) >= 2 and n >= 5:
            for i in range(len(gain_list) - 1):
                g_a, g_b = gain_list[i], gain_list[i + 1]
                (sz_a1, sz_a2), (sz_b1, sz_b2) = gain_keys[i], gain_keys[i + 1]
                diff_gains = g_a - g_b
                stat, p = stats.wilcoxon(diff_gains, alternative='two-sided')
                r = 1 - (2 * stat) / (n * (n + 1) / 2)
                n_first_bigger = (diff_gains > 0).sum()
                label_a = f"{size_labels[sz_a1]}→{size_labels[sz_a2]}"
                label_b = f"{size_labels[sz_b1]}→{size_labels[sz_b2]}"
                print(f"  Ratio ({label_a})/({label_b}): {g_a.mean() / max(g_b.mean(), 1e-6):.2f}x")
                print(f"  Wilcoxon ({label_a} gain vs {label_b} gain): p={p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
                print(f"  Effect size (r): {r:.4f}")
                print(f"  Sessions with larger {label_a} gain: {n_first_bigger}/{n} ({n_first_bigger/n*100:.1f}%)")

                wilcoxon_results.append({
                    'comparison': f'{label_a}_gain_vs_{label_b}_gain',
                    'n': n,
                    'mean_diff': diff_gains.mean(),
                    'std_diff': diff_gains.std(),
                    'wilcoxon_stat': stat,
                    'p_value': p,
                    'effect_size_r': r,
                    'n_improved': n_first_bigger,
                    'pct_improved': n_first_bigger/n*100
                })

    # Save results
    results_df = pd.DataFrame(wilcoxon_results)
    results_df.to_csv(OUTPUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'wilcoxon_results.csv'}")

    # Create visualization
    create_wilcoxon_figure(session_stats, results_df)

    return results_df


def variance_comparison_analysis(session_stats, n_resamples=10000, random_state=42):
    """Three complementary tests that TTA has lower variance than vanilla.

    1. Brown-Forsythe (one-sided): H1: var(vanilla) > var(TTA)
       Robust to non-normality; one-sided p = two-sided p / 2 when van.std() > tta.std().

    2. Bootstrap CI on SD ratio (vanilla SD / TTA SD):
       Resample paired sessions; CI entirely above 1.0 confirms ratio > 1.

    3. Pitman-Morgan test:
       Designed for paired (correlated) data. Tests H0: var(TTA) == var(vanilla)
       via correlation of (TTA+vanilla) with (TTA-vanilla).
       One-sided p = two-sided p / 2 when van.std() > tta.std().
    """
    print("\n" + "=" * 60)
    print("VARIANCE COMPARISON: TTA vs VANILLA")
    print("H1 (all tests): var(vanilla) > var(TTA)")
    print("=" * 60)

    rng = np.random.default_rng(random_state)
    support_sizes = sorted(session_stats['support_samples'].unique())
    results = []

    for sz in support_sizes:
        subset = session_stats[session_stats['support_samples'] == sz].dropna(
            subset=['tta_r2_mean', 'vanilla_test_r2'])
        tta = subset['tta_r2_mean'].values
        van = subset['vanilla_test_r2'].values
        n = len(tta)

        # --- 1. Brown-Forsythe (one-sided) ---
        bf_stat, bf_p_two = stats.levene(tta, van, center='median')
        bf_p_one = bf_p_two / 2 if van.std() > tta.std() else 1 - bf_p_two / 2

        # --- 2. Bootstrap CI on SD ratio ---
        ratios = []
        for _ in range(n_resamples):
            idx = rng.integers(n, size=n)
            ratios.append(van[idx].std() / tta[idx].std())
        ratio_obs = van.std() / tta.std()
        ratio_ci_lo = np.percentile(ratios, 2.5)
        ratio_ci_hi = np.percentile(ratios, 97.5)

        # --- 3. Pitman-Morgan (one-sided) ---
        sums = tta + van
        diffs = tta - van
        pm_r, pm_p_two = stats.pearsonr(sums, diffs)
        pm_p_one = pm_p_two / 2 if van.std() > tta.std() else 1 - pm_p_two / 2

        def sig(p):
            return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

        print(f"\n{sz} support (n={n}):")
        print(f"  TTA    SD = {tta.std():.4f}")
        print(f"  Vanilla SD = {van.std():.4f}")
        print(f"  Brown-Forsythe:  F={bf_stat:.4f}  p(one-sided)={bf_p_one:.4f}  {sig(bf_p_one)}")
        print(f"  SD ratio (van/TTA): {ratio_obs:.3f}x  95% CI [{ratio_ci_lo:.3f}, {ratio_ci_hi:.3f}]"
              f"  {'CI > 1.0' if ratio_ci_lo > 1.0 else 'CI includes 1.0'}")
        print(f"  Pitman-Morgan:   r={pm_r:.4f}  p(one-sided)={pm_p_one:.4f}  {sig(pm_p_one)}")

        results.append({
            'support_samples': sz,
            'tta_sd': tta.std(),
            'vanilla_sd': van.std(),
            'sd_ratio': ratio_obs,
            'sd_ratio_ci_lo': ratio_ci_lo,
            'sd_ratio_ci_hi': ratio_ci_hi,
            'bf_stat': bf_stat,
            'bf_p_one_sided': bf_p_one,
            'pitman_morgan_r': pm_r,
            'pitman_morgan_p_one_sided': pm_p_one,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "variance_comparison_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'variance_comparison_results.csv'}")
    return df


def create_wilcoxon_figure(session_stats, wilcoxon_results):
    """Create visualization of Wilcoxon analysis results."""

    fig_dir = OUTPUT_DIR / "figures"
    support_sizes = sorted(session_stats['support_samples'].unique())

    support_colors = {500: '#9b59b6', 1000: '#e74c3c', 2500: '#f39c12', 5000: '#3498db'}
    support_labels = {500: '500', 1000: '1K', 2500: '2.5K', 5000: '5K'}

    # Build per-size indexed stats
    stats_by_size = {}
    for sz in support_sizes:
        stats_by_size[sz] = session_stats[session_stats['support_samples'] == sz].set_index('session_id')

    fig = plt.figure(figsize=(18, 12))

    # 1. TTA R² boxplot by support size
    ax1 = fig.add_subplot(2, 3, 1)
    box_data = []
    box_labels = []
    box_colors_list = []
    for sz in support_sizes:
        vals = stats_by_size[sz]['tta_r2_mean'].dropna().values
        box_data.append(vals)
        box_labels.append(support_labels.get(sz, str(sz)))
        box_colors_list.append(support_colors.get(sz, '#333'))

    bp = ax1.boxplot(box_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticklabels(box_labels)
    ax1.set_ylabel('TTA R² (session mean)', fontsize=11)
    ax1.set_title('TTA Performance by Support Size', fontsize=12, fontweight='bold')

    # 2. Pairwise scatter: smallest vs largest support
    ax2 = fig.add_subplot(2, 3, 2)
    sz_min, sz_max = min(support_sizes), max(support_sizes)
    shared = stats_by_size[sz_min].index.intersection(stats_by_size[sz_max].index)
    tta_lo = stats_by_size[sz_min].loc[shared, 'tta_r2_mean'].values
    tta_hi = stats_by_size[sz_max].loc[shared, 'tta_r2_mean'].values

    ax2.scatter(tta_lo, tta_hi, alpha=0.6, s=60, c='#2c3e50', edgecolors='white', linewidth=0.5)
    lims = [min(tta_lo.min(), tta_hi.min()) - 0.05, max(tta_lo.max(), tta_hi.max()) + 0.05]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(f'{support_labels[sz_min]} TTA R²', fontsize=11)
    ax2.set_ylabel(f'{support_labels[sz_max]} TTA R²', fontsize=11)
    ax2.set_title(f'{support_labels[sz_max]} vs {support_labels[sz_min]} TTA\n(paired by session)', fontsize=12, fontweight='bold')
    n_hi_better = (tta_hi > tta_lo).sum()
    ax2.text(0.05, 0.95, f'{support_labels[sz_max]} > {support_labels[sz_min]}: {n_hi_better}/{len(shared)}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Effect sizes for all comparisons (filter out NaN rows)
    ax3 = fig.add_subplot(2, 3, 3)
    valid_results = wilcoxon_results.dropna(subset=['effect_size_r', 'p_value'])
    comp_labels = [r['comparison'].replace('_', ' ')[:20] for _, r in valid_results.iterrows()]
    effect_sizes = valid_results['effect_size_r'].values
    p_values = valid_results['p_value'].values

    palette = sns.color_palette("husl", len(comp_labels))
    bars = ax3.barh(range(len(comp_labels)), effect_sizes, color=palette, alpha=0.7, edgecolor='black')
    ax3.set_yticks(range(len(comp_labels)))
    ax3.set_yticklabels(comp_labels, fontsize=7)

    for i, (bar, p) in enumerate(zip(bars, p_values)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, sig,
                va='center', fontsize=9, fontweight='bold')

    ax3.axvline(0.5, color='green', linestyle=':', alpha=0.7, label='Large (0.5)')
    ax3.axvline(0.3, color='orange', linestyle=':', alpha=0.7, label='Medium (0.3)')
    ax3.set_xlabel('Effect Size (r)', fontsize=11)
    ax3.set_title('Wilcoxon Effect Sizes', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.1)
    ax3.legend(loc='lower right', fontsize=8)

    # 4. Diminishing returns: per-session gain at each step
    ax4 = fig.add_subplot(2, 3, 4)
    if len(support_sizes) >= 3:
        all_three = stats_by_size[support_sizes[0]].index
        for sz in support_sizes[1:]:
            all_three = all_three.intersection(stats_by_size[sz].index)

        gains = {}
        for i in range(len(support_sizes) - 1):
            sz_a, sz_b = support_sizes[i], support_sizes[i+1]
            g = stats_by_size[sz_b].loc[all_three, 'tta_r2_mean'].values - \
                stats_by_size[sz_a].loc[all_three, 'tta_r2_mean'].values
            label = f'{support_labels[sz_a]}→{support_labels[sz_b]}'
            gains[label] = g

        bp4 = ax4.boxplot(list(gains.values()), patch_artist=True, widths=0.6)
        step_colors = ['#27ae60', '#2980b9', '#8e44ad']
        for patch, color in zip(bp4['boxes'], step_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_xticklabels(list(gains.keys()))
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Marginal R² Gain', fontsize=11)
        ax4.set_title('Diminishing Returns Analysis', fontsize=12, fontweight='bold')

    # 5. Summary statistics table
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')

    table_data = []
    for _, row in wilcoxon_results.dropna(subset=['effect_size_r', 'p_value']).iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        effect = 'large' if abs(row['effect_size_r']) > 0.5 else 'medium' if abs(row['effect_size_r']) > 0.3 else 'small'
        table_data.append([
            row['comparison'].replace('_', ' ')[:25],
            f"{row['mean_diff']:.4f}",
            f"{row['p_value']:.2e}{sig}",
            f"{row['effect_size_r']:.3f} ({effect})"
        ])

    if table_data:
        table = ax5.table(
            cellText=table_data,
            colLabels=['Comparison', 'Mean Diff', 'p-value', 'Effect Size'],
            loc='center',
            cellLoc='center',
            colWidths=[0.35, 0.18, 0.22, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.6)
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax5.set_title('Wilcoxon Test Results Summary', fontsize=12, fontweight='bold', pad=20)

    # 6. Per-session R² across all support sizes (line plot)
    ax6 = fig.add_subplot(2, 3, 6)
    all_sessions_idx = stats_by_size[support_sizes[0]].index
    for sz in support_sizes[1:]:
        all_sessions_idx = all_sessions_idx.intersection(stats_by_size[sz].index)

    if len(all_sessions_idx) > 0:
        for session in sorted(all_sessions_idx):
            vals = [stats_by_size[sz].loc[session, 'tta_r2_mean'] for sz in support_sizes]
            ax6.plot(range(len(support_sizes)), vals, 'o-', alpha=0.3, markersize=3, color='gray')

        # Mean line
        mean_vals = [stats_by_size[sz].loc[all_sessions_idx, 'tta_r2_mean'].mean() for sz in support_sizes]
        ax6.plot(range(len(support_sizes)), mean_vals, 'ko-', markersize=8, linewidth=2.5, label='Mean', zorder=5)

        ax6.set_xticks(range(len(support_sizes)))
        ax6.set_xticklabels([support_labels[sz] for sz in support_sizes])
        ax6.set_xlabel('Support Samples', fontsize=11)
        ax6.set_ylabel('TTA R² (session mean)', fontsize=11)
        ax6.set_title('Per-Session TTA Trajectory', fontsize=12, fontweight='bold')
        ax6.legend()

    support_str = ', '.join([support_labels.get(s, str(s)) for s in support_sizes])
    plt.suptitle(f'Wilcoxon Signed-Rank Analysis: {support_str} Support Samples',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_dir / "wilcoxon_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'wilcoxon_analysis.png'}")


def bootstrap_prediction_interval(vals, confidence=0.95, n_resamples=10000, random_state=42):
    """Bootstrap prediction interval for a single new randomly drawn observation.

    Combines two sources of variability:
      1. Uncertainty in the population mean (from resampling)
      2. Natural spread of individual observations around the mean (from residuals)

    Each bootstrap iteration: draw n with replacement -> compute resample mean ->
    add a random residual from the original data -> that's a simulated new observation.
    The CI on those simulated observations is the prediction interval.
    """
    rng = np.random.default_rng(random_state)
    n = len(vals)
    residuals = vals - vals.mean()
    predictions = []
    for _ in range(n_resamples):
        resample = rng.choice(vals, size=n, replace=True)
        new_obs = resample.mean() + rng.choice(residuals)
        predictions.append(new_obs)
    alpha = 1 - confidence
    lo = np.percentile(predictions, 100 * alpha / 2)
    hi = np.percentile(predictions, 100 * (1 - alpha / 2))
    return lo, hi


def bootstrap_delta_prediction_interval(tta_vals, van_vals, confidence=0.95, n_resamples=10000, random_state=42):
    """Bootstrap prediction interval on paired TTA - vanilla improvement for a new session.

    Pairs sessions before bootstrapping to remove session-level noise common to both
    methods.  The resulting PI covers where a new (unseen) session's improvement will
    land, not just the uncertainty in the mean improvement.

    Each bootstrap iteration:
      1. Resample n paired deltas with replacement -> compute resample mean
      2. Add one random residual from the original 40 deltas -> simulated new session delta
    The 2.5th / 97.5th percentiles of those simulated deltas form the 95% PI.
    """
    rng = np.random.default_rng(random_state)
    deltas = tta_vals - van_vals
    n = len(deltas)
    residuals = deltas - deltas.mean()
    predictions = []
    for _ in range(n_resamples):
        resample = rng.choice(deltas, size=n, replace=True)
        new_obs = resample.mean() + rng.choice(residuals)
        predictions.append(new_obs)
    alpha = 1 - confidence
    lo = np.percentile(predictions, 100 * alpha / 2)
    hi = np.percentile(predictions, 100 * (1 - alpha / 2))
    return lo, hi


def create_delta_pi_figure(session_stats):
    """Plot bootstrap PI on TTA - vanilla improvement per support size."""

    fig_dir = OUTPUT_DIR / "figures"
    support_sizes = sorted(session_stats['support_samples'].unique())
    support_labels = {500: '500', 1000: '1K', 2500: '2.5K', 5000: '5K'}

    delta_means, delta_ci_lo, delta_ci_hi = [], [], []
    pi_lo_vals, pi_hi_vals = [], []

    for sz in support_sizes:
        subset = session_stats[session_stats['support_samples'] == sz].dropna(
            subset=['tta_r2_mean', 'vanilla_test_r2'])
        tta_vals = subset['tta_r2_mean'].values
        van_vals = subset['vanilla_test_r2'].values
        deltas = tta_vals - van_vals

        delta_means.append(deltas.mean())

        ci = stats.bootstrap((deltas,), np.mean, confidence_level=0.95,
                             n_resamples=10000, random_state=42)
        delta_ci_lo.append(deltas.mean() - ci.confidence_interval.low)
        delta_ci_hi.append(ci.confidence_interval.high - deltas.mean())

        lo, hi = bootstrap_delta_prediction_interval(tta_vals, van_vals)
        pi_lo_vals.append(lo)
        pi_hi_vals.append(hi)

    x = np.arange(len(support_sizes))
    labels = [support_labels.get(sz, str(sz)) for sz in support_sizes]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Prediction interval shaded band
    ax.fill_between(x, pi_lo_vals, pi_hi_vals, alpha=0.15, color='#9b59b6',
                    label='95% prediction interval (new session)')

    # Zero reference
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)

    # Mean + CI error bars
    ax.errorbar(x, delta_means,
                yerr=[delta_ci_lo, delta_ci_hi],
                fmt='o-', color='#9b59b6', linewidth=2, markersize=7,
                capsize=4, label='Mean improvement ± 95% CI')

    # Individual session dots
    for i, sz in enumerate(support_sizes):
        subset = session_stats[session_stats['support_samples'] == sz].dropna(
            subset=['tta_r2_mean', 'vanilla_test_r2'])
        deltas = subset['tta_r2_mean'].values - subset['vanilla_test_r2'].values
        ax.scatter(np.full(len(deltas), i), deltas, color='#9b59b6',
                   alpha=0.25, s=18, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel('Support Size', fontsize=13)
    ax.set_ylabel('R² Improvement (TTA − Vanilla)', fontsize=13)
    ax.set_title('TTA Improvement Over Vanilla\nMean ± 95% CI with Session-Level Prediction Interval',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(fig_dir / "delta_prediction_interval.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'delta_prediction_interval.png'}")


def create_learning_curve_figure(session_stats):
    """Create a learning curve: TTA R² and Vanilla R² vs support size as line plot."""

    fig_dir = OUTPUT_DIR / "figures"
    support_sizes = sorted(session_stats['support_samples'].unique())
    support_labels = {500: '500', 1000: '1K', 2500: '2.5K', 5000: '5K'}

    # Compute means, 95% bootstrap CIs (on mean), and prediction intervals (single session)
    tta_means, tta_ci_lo, tta_ci_hi = [], [], []
    tta_pi_lo, tta_pi_hi = [], []
    van_means, van_ci_lo, van_ci_hi = [], [], []
    van_pi_lo, van_pi_hi = [], []
    van_sizes_available = []

    for sz in support_sizes:
        subset = session_stats[session_stats['support_samples'] == sz]
        tta_vals = subset['tta_r2_mean'].dropna().values
        tta_mean = tta_vals.mean()
        tta_means.append(tta_mean)
        ci = stats.bootstrap((tta_vals,), np.mean, confidence_level=0.95,
                             n_resamples=10000, random_state=42)
        tta_ci_lo.append(tta_mean - ci.confidence_interval.low)
        tta_ci_hi.append(ci.confidence_interval.high - tta_mean)
        pi_lo, pi_hi = bootstrap_prediction_interval(tta_vals)
        tta_pi_lo.append(pi_lo)
        tta_pi_hi.append(pi_hi)

        van_vals = subset['vanilla_test_r2'].dropna().values
        if len(van_vals) > 0:
            van_mean = van_vals.mean()
            van_means.append(van_mean)
            ci = stats.bootstrap((van_vals,), np.mean, confidence_level=0.95,
                                 n_resamples=10000, random_state=42)
            van_ci_lo.append(van_mean - ci.confidence_interval.low)
            van_ci_hi.append(ci.confidence_interval.high - van_mean)
            pi_lo, pi_hi = bootstrap_prediction_interval(van_vals)
            van_pi_lo.append(pi_lo)
            van_pi_hi.append(pi_hi)
            van_sizes_available.append(sz)

    x_tta = np.arange(len(support_sizes))
    x_van = np.array([support_sizes.index(sz) for sz in van_sizes_available])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Prediction interval bands (wide, faint) — where a new session would land
    ax.fill_between(x_tta, tta_pi_lo, tta_pi_hi, alpha=0.08, color='#2ecc71',
                    label='95% prediction interval (new session)')
    if len(van_sizes_available) > 0:
        ax.fill_between(x_van, van_pi_lo, van_pi_hi, alpha=0.08, color='#e74c3c')

    # Vanilla line with CI error bars
    ax.errorbar(x_van, van_means, yerr=[van_ci_lo, van_ci_hi], fmt='s-',
                color='#e74c3c', markersize=9, linewidth=2, capsize=5,
                capthick=1.5, label='Vanilla (single-session)',
                markerfacecolor='white', markeredgewidth=2, zorder=3)

    # TTA line with CI error bars
    ax.errorbar(x_tta, tta_means, yerr=[tta_ci_lo, tta_ci_hi], fmt='o-',
                color='#2ecc71', markersize=9, linewidth=2, capsize=5,
                capthick=1.5, label='MAML + TTA', markerfacecolor='white',
                markeredgewidth=2, zorder=3)

    # Shade the gap between curves at shared points
    if len(van_sizes_available) == len(support_sizes):
        ax.fill_between(x_tta, van_means, tta_means, alpha=0.12, color='#2ecc71')

    # Annotate the MAML advantage at each shared point
    for i, sz in enumerate(van_sizes_available):
        idx = support_sizes.index(sz)
        diff = tta_means[idx] - van_means[i]
        mid_y = (tta_means[idx] + van_means[i]) / 2
        sign = '+' if diff > 0 else ''
        ax.annotate(f'{sign}{diff:.3f}', xy=(idx + 0.08, mid_y),
                    fontsize=9, color='#27ae60', fontweight='bold')

    ax.set_xticks(x_tta)
    ax.set_xticklabels([support_labels.get(sz, str(sz)) for sz in support_sizes], fontsize=12)
    ax.set_xlabel('Support Size', fontsize=13)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title('Learning Curve: MAML+TTA vs Vanilla', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curve.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'learning_curve.png'}")


def create_spaghetti_figure(session_stats):
    """Create mean ± SD trajectory plot with vanilla overlay."""

    fig_dir = OUTPUT_DIR / "figures"
    support_sizes = sorted(session_stats['support_samples'].unique())
    support_labels = {500: '500', 1000: '1K', 2500: '2.5K', 5000: '5K'}

    # Build per-size indexed stats
    stats_by_size = {}
    for sz in support_sizes:
        stats_by_size[sz] = session_stats[session_stats['support_samples'] == sz].set_index('session_id')

    # Sessions present at all support sizes
    all_sessions = stats_by_size[support_sizes[0]].index
    for sz in support_sizes[1:]:
        all_sessions = all_sessions.intersection(stats_by_size[sz].index)

    if len(all_sessions) == 0:
        print("WARNING: No sessions shared across all support sizes for trajectory plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(support_sizes))

    # TTA: mean ± SD
    tta_means = [stats_by_size[sz].loc[all_sessions, 'tta_r2_mean'].mean() for sz in support_sizes]
    tta_stds = [stats_by_size[sz].loc[all_sessions, 'tta_r2_mean'].std() for sz in support_sizes]
    tta_means_arr = np.array(tta_means)
    tta_stds_arr = np.array(tta_stds)

    ax.fill_between(x, tta_means_arr - tta_stds_arr, tta_means_arr + tta_stds_arr,
                     alpha=0.15, color='#2ecc71')
    ax.plot(x, tta_means, 'o-', color='#27ae60', markersize=10, linewidth=3,
            label='MAML+TTA (mean ± SD)', zorder=5, markerfacecolor='white',
            markeredgewidth=2.5)

    # Vanilla: mean ± SD
    van_means, van_stds = [], []
    van_x = []
    for i, sz in enumerate(support_sizes):
        van_col = stats_by_size[sz].loc[all_sessions, 'vanilla_test_r2'].dropna()
        if len(van_col) > 0:
            van_means.append(van_col.mean())
            van_stds.append(van_col.std())
            van_x.append(i)

    if van_x:
        van_means_arr = np.array(van_means)
        van_stds_arr = np.array(van_stds)
        van_x_arr = np.array(van_x)
        ax.fill_between(van_x_arr, van_means_arr - van_stds_arr,
                         van_means_arr + van_stds_arr, alpha=0.15, color='#e74c3c')
        ax.plot(van_x_arr, van_means_arr, 's-', color='#c0392b', markersize=10,
                linewidth=3, label='Vanilla (mean ± SD)', zorder=5,
                markerfacecolor='white', markeredgewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels([support_labels.get(sz, str(sz)) for sz in support_sizes], fontsize=12)
    ax.set_xlabel('Support Size', fontsize=13)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title('Session Variability: MAML+TTA vs Vanilla', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.02, f'n = {len(all_sessions)} sessions',
            transform=ax.transAxes, fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(fig_dir / "spaghetti_trajectories.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'spaghetti_trajectories.png'}")


def create_violin_figure(session_stats):
    """Create violin plot showing R² distributions at each support size."""

    fig_dir = OUTPUT_DIR / "figures"
    support_sizes = sorted(session_stats['support_samples'].unique())
    support_labels = {500: '500', 1000: '1K', 2500: '2.5K', 5000: '5K'}

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = np.arange(len(support_sizes))

    # Collect data for violins
    tta_data = []
    van_data = []
    van_positions = []

    for i, sz in enumerate(support_sizes):
        subset = session_stats[session_stats['support_samples'] == sz]
        tta_data.append(subset['tta_r2_mean'].dropna().values)

        van_vals = subset['vanilla_test_r2'].dropna().values
        if len(van_vals) > 0:
            van_data.append(van_vals)
            van_positions.append(i)

    # Vanilla violins (left side)
    if van_data:
        vp_van = ax.violinplot(van_data, positions=np.array(van_positions) - width / 2,
                               widths=width, showextrema=False)
        for body in vp_van['bodies']:
            body.set_facecolor('#e74c3c')
            body.set_alpha(0.4)
        # Overlay box summary (quartiles + median)
        bp_van = ax.boxplot(van_data, positions=np.array(van_positions) - width / 2,
                            widths=width * 0.4, patch_artist=True,
                            showfliers=False, zorder=3)
        for patch in bp_van['boxes']:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.6)
        for median in bp_van['medians']:
            median.set_color('white')
            median.set_linewidth(2)

    # TTA violins (right side)
    vp_tta = ax.violinplot(tta_data, positions=x + width / 2,
                           widths=width, showextrema=False)
    for body in vp_tta['bodies']:
        body.set_facecolor('#2ecc71')
        body.set_alpha(0.4)
    bp_tta = ax.boxplot(tta_data, positions=x + width / 2,
                        widths=width * 0.4, patch_artist=True,
                        showfliers=False, zorder=3)
    for patch in bp_tta['boxes']:
        patch.set_facecolor('#2ecc71')
        patch.set_alpha(0.6)
    for median in bp_tta['medians']:
        median.set_color('white')
        median.set_linewidth(2)

    # Legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#e74c3c', alpha=0.5, label='Vanilla'),
                       Patch(facecolor='#2ecc71', alpha=0.5, label='MAML + TTA')],
              fontsize=11, loc='lower right')

    ax.set_xticks(x)
    ax.set_xticklabels([support_labels.get(sz, str(sz)) for sz in support_sizes], fontsize=12)
    ax.set_xlabel('Support Size', fontsize=13)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title('R² Distribution by Support Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(fig_dir / "violin_distributions.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_dir / 'violin_distributions.png'}")


def main():
    print("=" * 60)
    print("Cross-Validation Results Analysis")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Support sample configurations: {list(TTA_CONFIGS.keys())}")
    print()

    # Parse data
    print("Parsing vanilla baseline results...")
    vanilla_df = parse_vanilla_results()
    print(f"  Total: {len(vanilla_df)} session-config pairs")

    print("\nParsing TTA results...")
    tta_df = parse_tta_results()
    print(f"  Total: {len(tta_df)} TTA evaluations")

    # Compute statistics
    print("\nComputing session statistics...")
    session_stats = compute_session_statistics(vanilla_df, tta_df)

    print("Computing fold statistics...")
    fold_stats = compute_fold_statistics(tta_df)

    # Generate outputs
    print("\n" + "-" * 40)
    print("Generating CSV files...")
    print("-" * 40)
    full_tta = create_comprehensive_csv(session_stats, tta_df, fold_stats, vanilla_df)

    print("\n" + "-" * 40)
    print("Generating visualizations...")
    print("-" * 40)
    create_visualizations(session_stats, tta_df, fold_stats, vanilla_df)

    # Publication figures
    print("\n" + "-" * 40)
    print("Generating publication figures...")
    print("-" * 40)
    create_learning_curve_figure(session_stats)
    create_spaghetti_figure(session_stats)
    create_violin_figure(session_stats)
    create_delta_pi_figure(session_stats)

    # Bootstrap CI and PI summary (for documentation)
    print("\n" + "=" * 60)
    print("BOOTSTRAP CI AND PREDICTION INTERVAL SUMMARY")
    print("=" * 60)
    support_sizes_pub = sorted(session_stats['support_samples'].unique())
    for sz in support_sizes_pub:
        subset = session_stats[session_stats['support_samples'] == sz]
        tta_vals = subset['tta_r2_mean'].dropna().values
        van_vals = subset['vanilla_test_r2'].dropna().values
        n = len(tta_vals)

        tta_mean = tta_vals.mean()
        tta_sd   = tta_vals.std()
        ci_tta = stats.bootstrap((tta_vals,), np.mean, confidence_level=0.95,
                                 n_resamples=10000, random_state=42)
        tta_ci = (ci_tta.confidence_interval.low, ci_tta.confidence_interval.high)
        tta_pi = bootstrap_prediction_interval(tta_vals)

        van_mean = van_vals.mean()
        van_sd   = van_vals.std()
        ci_van = stats.bootstrap((van_vals,), np.mean, confidence_level=0.95,
                                 n_resamples=10000, random_state=42)
        van_ci = (ci_van.confidence_interval.low, ci_van.confidence_interval.high)
        van_pi = bootstrap_prediction_interval(van_vals)

        print(f"\n{sz} support (n={n}):")
        print(f"  TTA  mean={tta_mean:.4f}  SD={tta_sd:.4f}")
        print(f"       95% CI  = [{tta_ci[0]:.4f}, {tta_ci[1]:.4f}]  (width={tta_ci[1]-tta_ci[0]:.4f})")
        print(f"       95% PI  = [{tta_pi[0]:.4f}, {tta_pi[1]:.4f}]  (width={tta_pi[1]-tta_pi[0]:.4f})")
        print(f"  Van  mean={van_mean:.4f}  SD={van_sd:.4f}")
        print(f"       95% CI  = [{van_ci[0]:.4f}, {van_ci[1]:.4f}]  (width={van_ci[1]-van_ci[0]:.4f})")
        print(f"       95% PI  = [{van_pi[0]:.4f}, {van_pi[1]:.4f}]  (width={van_pi[1]-van_pi[0]:.4f})")

        deltas = tta_vals - van_vals
        delta_mean = deltas.mean()
        delta_sd = deltas.std()
        ci_delta = stats.bootstrap((deltas,), np.mean, confidence_level=0.95,
                                   n_resamples=10000, random_state=42)
        delta_ci = (ci_delta.confidence_interval.low, ci_delta.confidence_interval.high)
        delta_pi = bootstrap_delta_prediction_interval(tta_vals, van_vals)
        p_positive = (deltas > 0).mean()
        print(f"  Delta (TTA-Van) mean={delta_mean:.4f}  SD={delta_sd:.4f}")
        print(f"       95% CI  = [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]  (width={delta_ci[1]-delta_ci[0]:.4f})")
        print(f"       95% PI  = [{delta_pi[0]:.4f}, {delta_pi[1]:.4f}]  (width={delta_pi[1]-delta_pi[0]:.4f})")
        print(f"       P(new session improves) = {p_positive:.2f}  ({int(p_positive*n)}/{n} sessions)")

    # Wilcoxon analysis
    wilcoxon_df = wilcoxon_analysis(session_stats, tta_df, vanilla_df)

    # Variance comparison (Brown-Forsythe, bootstrap SD ratio CI, Pitman-Morgan)
    variance_comparison_analysis(session_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    support_sizes = sorted(tta_df['support_samples'].unique())

    for n_support in support_sizes:
        tta_subset = tta_df[tta_df['support_samples'] == n_support]
        vanilla_subset = vanilla_df[vanilla_df['support_samples'] == n_support]
        session_subset = session_stats[session_stats['support_samples'] == n_support]
        fold_subset = fold_stats[fold_stats['support_samples'] == n_support]

        n_sessions = len(session_subset)
        n_improved = (session_subset['improvement_vs_vanilla_test'] > 0).sum()

        print(f"\n{'─' * 40}")
        print(f"SUPPORT SAMPLES: {n_support}")
        print(f"{'─' * 40}")
        print(f"Sessions: {n_sessions}")
        print(f"Folds: {len(fold_subset)}")
        print(f"TTA Evaluations: {len(tta_subset)}")
        print(f"\nTTA Performance:")
        print(f"  Mean R²: {tta_subset['tta_r2'].mean():.4f} ± {tta_subset['tta_r2'].std():.4f}")
        print(f"  Range: [{tta_subset['tta_r2'].min():.4f}, {tta_subset['tta_r2'].max():.4f}]")
        if not vanilla_subset.empty:
            print(f"\nVanilla Baseline:")
            print(f"  Mean Test R²: {vanilla_subset['vanilla_test_r2'].mean():.4f} ± {vanilla_subset['vanilla_test_r2'].std():.4f}")
            imp_col = session_subset['improvement_vs_vanilla_test'].dropna()
            if len(imp_col) > 0:
                print(f"\nImprovement vs Vanilla:")
                print(f"  Mean: {imp_col.mean():.4f}")
                print(f"  Sessions Improved: {n_improved}/{len(imp_col)} ({n_improved/len(imp_col)*100:.1f}%)")
        else:
            print(f"\nVanilla Baseline: N/A (no vanilla data for this support size)")

    print(f"\n{'=' * 60}")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
