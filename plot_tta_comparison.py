#!/usr/bin/env python3
"""
Compare TTA results from multiple JSON files on the same graph.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path):
    """Load results from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_combined_results(json_files, output_path=None):
    """
    Plot results from multiple JSON files on the same graph.
    
    Args:
        json_files: List of paths to JSON result files
        output_path: Optional path to save the figure
    """
    results_list = [load_results(f) for f in json_files]
    
    # Get unique strategies and models across all files
    all_strategies = set()
    all_models = set()
    for results in results_list:
        all_strategies.update(results['strategies'].keys())
        all_models.update(results['grid'].keys())
    all_strategies = sorted(all_strategies)
    all_models = sorted(all_models)
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette for different models
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    model_color_map = {model: model_colors[i] for i, model in enumerate(all_models)}
    
    # Marker and line style for different runs
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    support_sizes = None
    
    # Plot each file's data
    for file_idx, (json_file, results) in enumerate(zip(json_files, results_list)):
        timestamp = results['metadata']['timestamp']
        file_label = f"Run {file_idx + 1} ({timestamp})"
        
        tta_comparison = results['grid']
        if support_sizes is None:
            support_sizes = results['metadata']['support_sizes']
        
        linestyle = line_styles[file_idx % len(line_styles)]
        marker = markers[file_idx % len(markers)]
        
        # Plot each strategy and model combination on the same graph
        for strategy_key in all_strategies:
            for model_key in sorted(tta_comparison.keys()):
                if strategy_key not in tta_comparison[model_key]:
                    continue
                
                data_points = sorted(
                    tta_comparison[model_key][strategy_key],
                    key=lambda x: x[0]
                )
                support_vals = [dp[0] for dp in data_points]
                r2_vals = [dp[1] for dp in data_points]
                
                strategy_label = results['strategies'].get(strategy_key, {}).get('label', strategy_key)
                label = f"{model_key} - {strategy_label} - {file_label}"
                
                ax.plot(
                    support_vals,
                    r2_vals,
                    marker=marker,
                    linewidth=2,
                    linestyle=linestyle,
                    color=model_color_map[model_key],
                    label=label,
                    alpha=0.8,
                    markersize=6,
                )
    
    # Configure the single axis
    ax.set_xlabel("Support Set Size", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_xscale("log")
    if support_sizes:
        ax.set_xticks(support_sizes)
        ax.set_xticklabels([str(s) for s in support_sizes], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    plt.title("TTA Performance Comparison Across Runs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare TTA results from multiple JSON files"
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="Paths to JSON result files to compare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the output plot"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    for f in args.json_files:
        if not f.exists():
            print(f"Error: File not found: {f}")
            return
    
    print(f"Comparing {len(args.json_files)} result files...")
    plot_combined_results(args.json_files, args.output)


if __name__ == "__main__":
    main()
