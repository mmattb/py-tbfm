import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/home/danmuir/GitHub/py-tbfm/session_count_sweep_20251203_222740/tta_results/tta_support_20251206_023945_per_session.csv')

print("="*80)
print("SESSION PERFORMANCE ANALYSIS - LOOKING FOR UNDERFITTING")
print("="*80)

# Basic statistics
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Extract model size from model name (e.g., '1kx5_maml' -> 5 sessions)
df['pretrain_sessions'] = df['model'].str.extract(r'1kx(\d+)_')[0].astype(int)

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(df['session_r2'].describe())

# Identify worst performing sessions
print("\n" + "="*80)
print("WORST PERFORMING SESSIONS (Bottom 10 by R²)")
print("="*80)
worst_sessions = df.nsmallest(10, 'session_r2')[['model', 'strategy', 'pretrain_sessions', 'session_id', 'session_r2', 'overall_r2']]
print(worst_sessions.to_string(index=False))

# Group by session to see which sessions consistently underperform
print("\n" + "="*80)
print("AVERAGE PERFORMANCE BY SESSION (across all models/strategies)")
print("="*80)
session_avg = df.groupby('session_id')['session_r2'].agg(['mean', 'std', 'min', 'max', 'count'])
session_avg = session_avg.sort_values('mean')
print(session_avg.to_string())

# Look for underfitting patterns: performance vs pretrain sessions
print("\n" + "="*80)
print("PERFORMANCE BY PRETRAIN SESSION COUNT (looking for underfitting)")
print("="*80)
pretrain_stats = df.groupby('pretrain_sessions')['session_r2'].agg(['mean', 'std', 'min', 'max'])
pretrain_stats = pretrain_stats.sort_index()
print(pretrain_stats.to_string())
print("\nNote: If performance improves significantly with more pretrain sessions,")
print("this suggests underfitting (model needs more data to learn).")

# Check if low pretrain sessions correlate with poor performance
print("\n" + "="*80)
print("UNDERFITTING ANALYSIS BY SESSION")
print("="*80)

# For each session, compare performance at low vs high pretrain counts
for session_id in session_avg.head(5).index:  # Focus on worst 5 sessions
    session_data = df[df['session_id'] == session_id].sort_values('pretrain_sessions')
    print(f"\n{session_id}:")
    print(f"  Pretrain sessions: {session_data['pretrain_sessions'].min()}-{session_data['pretrain_sessions'].max()}")
    print(f"  R² at min pretrain ({session_data['pretrain_sessions'].min()}): {session_data[session_data['pretrain_sessions'] == session_data['pretrain_sessions'].min()]['session_r2'].mean():.4f}")
    print(f"  R² at max pretrain ({session_data['pretrain_sessions'].max()}): {session_data[session_data['pretrain_sessions'] == session_data['pretrain_sessions'].max()]['session_r2'].mean():.4f}")
    improvement = session_data[session_data['pretrain_sessions'] == session_data['pretrain_sessions'].max()]['session_r2'].mean() - \
                  session_data[session_data['pretrain_sessions'] == session_data['pretrain_sessions'].min()]['session_r2'].mean()
    print(f"  Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
    if improvement > 0.1:
        print(f"  ⚠️  STRONG UNDERFITTING - Large improvement with more data")

# Strategy comparison for worst sessions
print("\n" + "="*80)
print("STRATEGY COMPARISON FOR WORST SESSIONS")
print("="*80)
worst_session_ids = session_avg.head(5).index.tolist()
strategy_comparison = df[df['session_id'].isin(worst_session_ids)].groupby(['session_id', 'strategy'])['session_r2'].mean().unstack()
print(strategy_comparison.to_string())

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution of session R² scores
ax1 = axes[0, 0]
ax1.hist(df['session_r2'], bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(df['session_r2'].mean(), color='red', linestyle='--', label=f'Mean: {df["session_r2"].mean():.3f}')
ax1.axvline(df['session_r2'].median(), color='green', linestyle='--', label=f'Median: {df["session_r2"].median():.3f}')
ax1.set_xlabel('Session R²')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Session R² Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Performance vs pretrain sessions (underfitting check)
ax2 = axes[0, 1]
for strategy in df['strategy'].unique():
    strategy_data = df[df['strategy'] == strategy].groupby('pretrain_sessions')['session_r2'].mean()
    ax2.plot(strategy_data.index, strategy_data.values, marker='o', label=strategy)
ax2.set_xlabel('Number of Pretrain Sessions')
ax2.set_ylabel('Average Session R²')
ax2.set_title('Performance vs Pretrain Sessions (Underfitting Check)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Boxplot by session (worst sessions highlighted)
ax3 = axes[1, 0]
session_r2_by_session = df.groupby('session_id')['session_r2'].apply(list).to_dict()
sessions_sorted = session_avg.sort_values('mean').index.tolist()
positions = range(len(sessions_sorted))
bp = ax3.boxplot([session_r2_by_session[s] for s in sessions_sorted],
                   positions=positions, patch_artist=True)

# Color worst 5 sessions red
for i, patch in enumerate(bp['boxes']):
    if i < 5:
        patch.set_facecolor('red')
        patch.set_alpha(0.5)
    else:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.5)

ax3.set_xlabel('Session')
ax3.set_ylabel('Session R²')
ax3.set_title('R² Distribution by Session (Red = Worst 5)')
ax3.set_xticks([])
ax3.grid(True, alpha=0.3)

# 4. Heatmap: pretrain sessions vs session performance
ax4 = axes[1, 1]
pivot_data = df.groupby(['session_id', 'pretrain_sessions'])['session_r2'].mean().unstack()
sns.heatmap(pivot_data.loc[sessions_sorted], annot=False, cmap='RdYlGn',
            vmin=0, vmax=0.7, ax=ax4, cbar_kws={'label': 'Session R²'})
ax4.set_xlabel('Number of Pretrain Sessions')
ax4.set_ylabel('Session')
ax4.set_title('R² Heatmap: Session vs Pretrain Size')

plt.tight_layout()
plt.savefig('/home/danmuir/GitHub/py-tbfm/session_performance_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n\nPlot saved to: /home/danmuir/GitHub/py-tbfm/session_performance_analysis.png")

# Summary conclusions
print("\n" + "="*80)
print("SUMMARY & CONCLUSIONS")
print("="*80)
print(f"1. Total unique sessions: {df['session_id'].nunique()}")
print(f"2. Performance range: {df['session_r2'].min():.4f} to {df['session_r2'].max():.4f}")
print(f"3. Worst performing session: {session_avg.index[0]} (avg R²: {session_avg.iloc[0]['mean']:.4f})")
print(f"4. Best performing session: {session_avg.index[-1]} (avg R²: {session_avg.iloc[-1]['mean']:.4f})")

# Calculate correlation between pretrain sessions and performance
correlation = df['pretrain_sessions'].corr(df['session_r2'])
print(f"\n5. Correlation between pretrain sessions and R²: {correlation:.4f}")
if correlation > 0.3:
    print("   ⚠️  POSITIVE CORRELATION suggests underfitting - more data helps significantly")
elif correlation > 0.1:
    print("   ⚠️  WEAK POSITIVE CORRELATION - mild underfitting present")
else:
    print("   ✓ Low correlation - underfitting may not be the main issue")

plt.show()
