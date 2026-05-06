#!/usr/bin/env python3
"""
Compare raw neural data characteristics between worst and best performing sessions.
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tbfm import multisession

# Constants
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")

# Worst vs best sessions from analysis
WORST_SESSIONS = [
    "MonkeyG_20150917_Session3_M1",  # R² = 0.084
    "MonkeyJ_20160429_Session3_S1",  # R² = 0.129
]

BEST_SESSIONS = [
    "MonkeyG_20150917_Session3_S1",  # R² = 0.648
    "MonkeyJ_20160627_Session1_S1",  # R² = 0.547
]


def analyze_session_data(session_id):
    """Load and analyze raw data for a session."""
    print(f"\nAnalyzing: {session_id}")

    # Load data
    try:
        d, _ = multisession.load_stim_batched(
            batch_size=7500,
            window_size=184,
            session_subdir="torchraw",
            data_dir=DATA_DIR,
            held_in_session_ids=[session_id],
            num_held_out_sessions=0,
        )
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

    data_train, data_test = d.train_test_split(5000, test_cut=2500)

    # Get a batch
    batch = next(iter(data_test))

    # The batch is structured by session ID
    if session_id in batch:
        session_batch = batch[session_id]
        # Session batch is a tuple: check what's in it
        if isinstance(session_batch, tuple):
            print(f"  Batch tuple length: {len(session_batch)}")
            for i, item in enumerate(session_batch):
                if hasattr(item, 'shape'):
                    print(f"    Element {i}: shape={item.shape}, dtype={item.dtype}")
            # The full neural data should be in one of these elements
            # Tuple is: (y_context, stim, y_target_full)
            if len(session_batch) >= 3:
                y_context = session_batch[0].cpu().numpy()  # Context/runway (batch, 20, channels)
                # Element 1 is stimulus
                y_target = session_batch[2].cpu().numpy()   # Full target trajectory (batch, 164, channels)
                print(f"  Context shape: {y_context.shape}, Target shape: {y_target.shape}")
                # Use full target trajectory as the main data
                y = y_target
            elif len(session_batch) >= 1:
                y = session_batch[0].cpu().numpy()
            else:
                print("  Error: Unexpected batch structure")
                return None
        else:
            y = session_batch.cpu().numpy()
    else:
        print(f"  Error: Session {session_id} not found in batch keys: {batch.keys()}")
        return None

    print(f"  Final data shape: y={y.shape}")

    # Compute statistics
    n_channels = y.shape[2]
    max_channels = 96  # Maximum possible channels per array
    n_dead_channels = max_channels - n_channels  # Already removed during preprocessing

    stats = {
        'session_id': session_id,
        'y_shape': y.shape,
        'n_channels': n_channels,
        'n_dead_channels': n_dead_channels,  # Channels already removed
        'y_mean': y.mean(),
        'y_std': y.std(),
        'y_min': y.min(),
        'y_max': y.max(),
        'y_range': y.max() - y.min(),
        'channel_means': y.mean(axis=(0, 1)),  # Mean per channel
        'channel_stds': y.std(axis=(0, 1)),    # Std per channel
        'channel_vars': y.var(axis=(0, 1)),    # Var per channel
        'temporal_autocorr': [],  # Will compute below
        'low_variance_channels': [],  # Channels with unusually low variance
        'high_variance_channels': [],  # Channels with unusually high variance
        'y_data': y,
    }

    # Identify channels with unusual variance (relative to other channels in this session)
    median_std = np.median(stats['channel_stds'])
    for ch in range(n_channels):
        ch_std = stats['channel_stds'][ch]
        # Low variance channel: less than 10% of median std
        if ch_std < 0.1 * median_std:
            stats['low_variance_channels'].append(ch)
        # High variance channel: more than 10x median std
        elif ch_std > 10 * median_std:
            stats['high_variance_channels'].append(ch)

    print(f"  Channels: {n_channels} active, {n_dead_channels} dead (removed in preprocessing)")
    print(f"  Mean: {stats['y_mean']:.4f}, Std: {stats['y_std']:.4f}")
    print(f"  Range: [{stats['y_min']:.4f}, {stats['y_max']:.4f}]")
    print(f"  Low variance channels ({len(stats['low_variance_channels'])}): {stats['low_variance_channels'][:10]}")
    print(f"  High variance channels ({len(stats['high_variance_channels'])}): {stats['high_variance_channels'][:10]}")
    print(f"  Median channel std: {np.median(stats['channel_stds']):.4f}")

    return stats


def visualize_comparison(worst_stats, best_stats):
    """Create comparative visualizations."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    all_sessions = worst_stats + best_stats
    colors = ['red'] * len(worst_stats) + ['green'] * len(best_stats)
    labels = ['WORST'] * len(worst_stats) + ['BEST'] * len(best_stats)

    # Plot 1: Overall distribution comparison
    ax1 = fig.add_subplot(gs[0, :2])
    for stats, color, label in zip(all_sessions, colors, labels):
        y_flat = stats['y_data'].flatten()
        ax1.hist(y_flat, bins=50, alpha=0.4, color=color, label=f"{label}: {stats['session_id'][:15]}")
    ax1.set_xlabel('Neural Activity')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Neural Activity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Channel variance comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    for stats, color, label in zip(all_sessions, colors, labels):
        ax2.plot(stats['channel_vars'], alpha=0.6, color=color, label=f"{label[:4]}: {stats['session_id'][:15]}")
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance by Channel')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean activity statistics
    ax3 = fig.add_subplot(gs[1, 0])
    session_names = [s['session_id'][:10] for s in all_sessions]
    means = [s['y_mean'] for s in all_sessions]
    ax3.bar(range(len(means)), means, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(means)))
    ax3.set_xticklabels(session_names, rotation=45, ha='right')
    ax3.set_ylabel('Mean Activity')
    ax3.set_title('Mean Neural Activity')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Std statistics
    ax4 = fig.add_subplot(gs[1, 1])
    stds = [s['y_std'] for s in all_sessions]
    ax4.bar(range(len(stds)), stds, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(stds)))
    ax4.set_xticklabels(session_names, rotation=45, ha='right')
    ax4.set_ylabel('Std Dev')
    ax4.set_title('Neural Activity Std Dev')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Dead channel count (pre-removed)
    ax5 = fig.add_subplot(gs[1, 2])
    dead_counts = [s['n_dead_channels'] for s in all_sessions]
    ax5.bar(range(len(dead_counts)), dead_counts, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(dead_counts)))
    ax5.set_xticklabels(session_names, rotation=45, ha='right')
    ax5.set_ylabel('Number of Dead Channels')
    ax5.set_title('Dead Channels (removed in preprocessing)')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Low variance channel count (among active channels)
    ax6 = fig.add_subplot(gs[1, 3])
    noisy_counts = [len(s['low_variance_channels']) for s in all_sessions]
    ax6.bar(range(len(noisy_counts)), noisy_counts, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(noisy_counts)))
    ax6.set_xticklabels(session_names, rotation=45, ha='right')
    ax6.set_ylabel('Number of Low-Var Channels')
    ax6.set_title('Low Variance Channels (std < 10% median)')
    ax6.grid(True, alpha=0.3)

    # Plot 7-8: Sample trajectories for worst session
    worst_example = worst_stats[0]
    ax7 = fig.add_subplot(gs[2, :2])
    n_samples = 5
    n_channels = 3
    for ch in range(n_channels):
        for i in range(n_samples):
            ax7.plot(worst_example['y_data'][i, :, ch], alpha=0.5, color=f'C{ch}')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Activity')
    ax7.set_title(f'WORST Session Trajectories: {worst_example["session_id"]}')
    ax7.grid(True, alpha=0.3)

    # Plot 9-10: Sample trajectories for best session
    best_example = best_stats[0]
    ax8 = fig.add_subplot(gs[2, 2:])
    for ch in range(n_channels):
        for i in range(n_samples):
            ax8.plot(best_example['y_data'][i, :, ch], alpha=0.5, color=f'C{ch}')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Activity')
    ax8.set_title(f'BEST Session Trajectories: {best_example["session_id"]}')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("/home/danmuir/GitHub/py-tbfm/session_data_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_path}")

    return fig


def main():
    """Main function."""
    print("="*80)
    print("COMPARING RAW DATA: WORST vs BEST SESSIONS")
    print("="*80)

    # Analyze worst sessions
    worst_stats = []
    print("\n" + "="*80)
    print("WORST PERFORMING SESSIONS")
    print("="*80)
    for session_id in WORST_SESSIONS:
        stats = analyze_session_data(session_id)
        if stats:
            worst_stats.append(stats)

    # Analyze best sessions
    best_stats = []
    print("\n" + "="*80)
    print("BEST PERFORMING SESSIONS")
    print("="*80)
    for session_id in BEST_SESSIONS:
        stats = analyze_session_data(session_id)
        if stats:
            best_stats.append(stats)

    # Create visualizations
    if worst_stats and best_stats:
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        visualize_comparison(worst_stats, best_stats)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    if worst_stats and best_stats:
        print("\nWORST SESSIONS:")
        for s in worst_stats:
            print(f"  {s['session_id']}:")
            print(f"    Channels: {s['n_channels']} active, {s['n_dead_channels']} dead")
            print(f"    Mean: {s['y_mean']:.4f}, Std: {s['y_std']:.4f}")
            print(f"    Low-variance channels: {len(s['low_variance_channels'])}")

        print("\nBEST SESSIONS:")
        for s in best_stats:
            print(f"  {s['session_id']}:")
            print(f"    Channels: {s['n_channels']} active, {s['n_dead_channels']} dead")
            print(f"    Mean: {s['y_mean']:.4f}, Std: {s['y_std']:.4f}")
            print(f"    Low-variance channels: {len(s['low_variance_channels'])}")

        # Key differences
        worst_dead_avg = np.mean([s['n_dead_channels'] for s in worst_stats])
        best_dead_avg = np.mean([s['n_dead_channels'] for s in best_stats])
        worst_std_avg = np.mean([s['y_std'] for s in worst_stats])
        best_std_avg = np.mean([s['y_std'] for s in best_stats])
        worst_channels_avg = np.mean([s['n_channels'] for s in worst_stats])
        best_channels_avg = np.mean([s['n_channels'] for s in best_stats])

        print("\nKEY DIFFERENCES:")
        print(f"  Average active channels: Worst={worst_channels_avg:.1f}, Best={best_channels_avg:.1f}")
        print(f"  Average dead channels (pre-removed): Worst={worst_dead_avg:.1f}, Best={best_dead_avg:.1f}")
        print(f"  Average std dev: Worst={worst_std_avg:.4f}, Best={best_std_avg:.4f}")

    print("\nAnalysis complete! Check session_data_comparison.png for visualizations.")


if __name__ == "__main__":
    main()
