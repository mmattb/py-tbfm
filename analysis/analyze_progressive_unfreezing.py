#!/usr/bin/env python3
"""
Analyze progressive unfreezing results.

Compares R² scores across different unfreezing strategies to determine
if supervised fine-tuning helps at high support sizes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_results(json_path: Path) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_comparison_data(results_dir: Path) -> pd.DataFrame:
    """
    Extract comparison data from all result files in directory structure.
    
    Expected structure:
        results_dir/
            baseline/tta_*.json
            unfreeze_basis_weights/tta_*.json
            unfreeze_bases/tta_*.json
            unfreeze_both/tta_*.json
    """
    data = []
    
    # Map directory names to strategy labels
    strategy_map = {
        'baseline': 'Baseline (frozen)',
        'unfreeze_basis_weights': 'Unfreeze basis weights',
        'unfreeze_bases': 'Unfreeze bases',
        'unfreeze_both': 'Unfreeze both',
    }
    
    for strategy_dir, strategy_label in strategy_map.items():
        strategy_path = results_dir / strategy_dir
        if not strategy_path.exists():
            continue
        
        # Find JSON files
        json_files = list(strategy_path.glob('tta_*.json'))
        if not json_files:
            continue
        
        # Load most recent file
        json_file = sorted(json_files)[-1]
        results = load_results(json_file)
        
        # Extract runs
        for run in results['runs']:
            data.append({
                'Strategy': strategy_label,
                'Model': run['model'],
                'Support Size': run['support_size'],
                'R²': run['r2'],
            })
    
    return pd.DataFrame(data)


def analyze_results(df: pd.DataFrame):
    """Analyze and print comparison results."""
    if len(df) == 0:
        print("No data found!")
        return
    
    print("=" * 80)
    print("Progressive Unfreezing Analysis")
    print("=" * 80)
    print()
    
    # Summary by strategy
    print("Average R² by Strategy:")
    print("-" * 80)
    strategy_avg = df.groupby('Strategy')['R²'].agg(['mean', 'std', 'count'])
    strategy_avg = strategy_avg.sort_values('mean', ascending=False)
    print(strategy_avg)
    print()
    
    # Summary by support size and strategy
    print("R² by Support Size and Strategy:")
    print("-" * 80)
    pivot = df.pivot_table(
        values='R²',
        index='Support Size',
        columns='Strategy',
        aggfunc='mean'
    )
    
    # Reorder columns if they exist
    col_order = ['Baseline (frozen)', 'Unfreeze basis weights', 
                 'Unfreeze bases', 'Unfreeze both']
    existing_cols = [c for c in col_order if c in pivot.columns]
    pivot = pivot[existing_cols]
    
    print(pivot)
    print()
    
    # Calculate improvements over baseline
    if 'Baseline (frozen)' in pivot.columns:
        print("Improvement over Baseline (R² points):")
        print("-" * 80)
        improvement = pivot.copy()
        baseline = improvement['Baseline (frozen)']
        
        for col in improvement.columns:
            if col != 'Baseline (frozen)':
                improvement[col] = improvement[col] - baseline
        
        improvement = improvement.drop('Baseline (frozen)', axis=1)
        print(improvement)
        print()
        
        # Identify best strategy per support size
        print("Best Strategy per Support Size:")
        print("-" * 80)
        best_strategies = pivot.idxmax(axis=1)
        best_r2 = pivot.max(axis=1)
        
        for support_size in sorted(pivot.index):
            best_strat = best_strategies[support_size]
            r2 = best_r2[support_size]
            baseline_r2 = baseline[support_size]
            improvement_val = r2 - baseline_r2
            
            print(f"Support {support_size:>5}: {best_strat:<25} "
                  f"(R²={r2:.4f}, +{improvement_val:+.4f} vs baseline)")
        print()
    
    # Statistical significance check (if we have multiple models/sessions)
    if df['Model'].nunique() > 1 or len(df[df['Strategy'] == 'Baseline (frozen)']) > len(df['Support Size'].unique()):
        print("Note: Multiple models/runs detected. Consider statistical testing.")
        print()
    
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_progressive_unfreezing.py <results_dir>")
        print()
        print("Example:")
        print("  python analyze_progressive_unfreezing.py data/tta_progressive_unfreezing")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"Loading results from: {results_dir}")
    print()
    
    df = extract_comparison_data(results_dir)
    
    if len(df) == 0:
        print("No results found in directory structure.")
        print("Expected structure:")
        print("  results_dir/")
        print("    baseline/tta_*.json")
        print("    unfreeze_basis_weights/tta_*.json")
        print("    unfreeze_bases/tta_*.json")
        print("    unfreeze_both/tta_*.json")
        sys.exit(1)
    
    analyze_results(df)
    
    # Save detailed results
    output_csv = results_dir / "progressive_unfreezing_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to: {output_csv}")


if __name__ == '__main__':
    main()
