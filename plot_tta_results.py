#!/usr/bin/env python3
"""
Plot TTA results from log files and/or JSON files on a unified graph.

This script reads TTA sweep results from both log files and JSON files,
combining them into a single comparison plot with session information.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: Path) -> Dict:
    """
    Parse a TTA sweep log file and extract results.

    Returns:
        Dictionary with structure:
        {
            'metadata': {
                'support_sizes': [...],
                'adapt_sessions': [...],
                'timestamp': str,
                'source': 'log'
            },
            'runs': [
                {'strategy': str, 'support_size': int, 'model': str, 'r2': float},
                ...
            ]
        }
    """
    results = {
        "metadata": {
            "support_sizes": [],
            "adapt_sessions": [],
            "timestamp": log_path.stem,
            "source": "log",
        },
        "runs": [],
    }

    with open(log_path, "r") as f:
        content = f.read()

    # Parse metadata from header
    support_match = re.search(r"Support sizes: (\[.*?\])", content)
    if support_match:
        support_str = support_match.group(1)
        results["metadata"]["support_sizes"] = sorted(
            set(int(x.strip()) for x in support_str.strip("[]").split(","))
        )

    session_match = re.search(r"Adapt session\(s\): (\[.*?\])", content)
    if session_match:
        session_str = session_match.group(1)
        # Parse list format string
        sessions = re.findall(r"'([^']+)'", session_str)
        results["metadata"]["adapt_sessions"] = sessions

    # Parse completion lines (lines with "R²=" in them)
    pattern = r"(?:Run complete|Complete)\s+\|\s+Strategy=([^|]+?)\s+\|\s+Support=\s*(\d+)\s+\|\s+Model=([^|]+?)\s+\|\s+R²=([+-]?\d+\.\d+)"

    for match in re.finditer(pattern, content):
        strategy = match.group(1).strip()
        support_size = int(match.group(2))
        model = match.group(3).strip()
        r2 = float(match.group(4))

        results["runs"].append(
            {
                "strategy": strategy,
                "support_size": support_size,
                "model": model,
                "r2": r2,
            }
        )

    return results


def parse_json_file(json_path: Path) -> Dict:
    """
    Parse a TTA sweep JSON file and extract results.

    Returns:
        Dictionary with same structure as parse_log_file for consistency.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {
        "metadata": {
            "support_sizes": data["metadata"].get("support_sizes", []),
            "adapt_sessions": data["metadata"].get("adapt_session_ids", 
                                                   data["metadata"].get("adapt_sessions", [])),
            "timestamp": data["metadata"].get("timestamp", json_path.stem),
            "source": "json",
        },
        "runs": data.get("runs", []),
    }

    return results


def load_result_file(file_path: Path) -> Dict:
    """
    Load results from either a log file or JSON file.
    
    Args:
        file_path: Path to log or JSON file
        
    Returns:
        Parsed results dictionary
    """
    if file_path.suffix == ".json":
        return parse_json_file(file_path)
    else:
        return parse_log_file(file_path)


def aggregate_runs_by_key(
    runs: List[Dict],
) -> Dict[Tuple[str, str], List[Tuple[int, float]]]:
    """
    Aggregate runs by (strategy, model) pairs.

    Returns:
        Dict mapping (strategy, model) to list of (support_size, r2) tuples
    """
    aggregated = {}

    for run in runs:
        key = (run["strategy"], run["model"])
        support_size = run["support_size"]
        r2 = run["r2"]

        if key not in aggregated:
            aggregated[key] = []

        aggregated[key].append((support_size, r2))

    # Sort by support size
    for key in aggregated:
        aggregated[key] = sorted(aggregated[key], key=lambda x: x[0])

    return aggregated


def plot_results(
    file_paths: List[Path],
    output_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Create comparison plots from log and/or JSON files.

    Args:
        file_paths: List of log/JSON file paths to plot
        output_path: Path to save plot. If None, uses timestamp-based name
        show: Whether to display plot interactively
    """
    # Parse all files
    all_results = []
    for file_path in file_paths:
        print(f"Parsing {file_path.name}...")
        results = load_result_file(file_path)
        all_results.append((file_path, results))

    if not all_results:
        print("No results found")
        return

    # Collect all unique strategies, models, and support sizes
    all_strategies = set()
    all_models = set()
    all_support_sizes = set()
    all_sessions = []

    for file_path, results in all_results:
        aggregated = aggregate_runs_by_key(results["runs"])
        
        for strategy, model in aggregated.keys():
            all_strategies.add(strategy)
            all_models.add(model)
        
        all_support_sizes.update(results["metadata"]["support_sizes"])
        
        # Collect session info
        if results["metadata"]["adapt_sessions"]:
            sessions_str = ", ".join(results["metadata"]["adapt_sessions"])
            all_sessions.append(f"{file_path.name}: {sessions_str}")

    all_strategies = sorted(all_strategies)
    all_models = sorted(all_models)
    support_sizes = sorted(all_support_sizes)

    # Create figure with extra space for session info
    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)

    # Use different line styles for files and markers/colors for strategy+model
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_strategies), len(all_models))))

    # Create a mapping for strategy+model combinations
    strategy_model_combos = []
    for strategy in all_strategies:
        for model in all_models:
            strategy_model_combos.append((strategy, model))

    combo_colors = plt.cm.tab20(np.linspace(0, 1, len(strategy_model_combos)))
    combo_to_color = {combo: combo_colors[i] for i, combo in enumerate(strategy_model_combos)}

    # Plot each file's data
    for file_idx, (file_path, results) in enumerate(all_results):
        aggregated = aggregate_runs_by_key(results["runs"])
        line_style = line_styles[file_idx % len(line_styles)]
        timestamp = results["metadata"]["timestamp"]
        file_label = f"Run {file_idx + 1} ({timestamp})"

        # Plot each strategy+model combination
        for combo_idx, (strategy, model) in enumerate(strategy_model_combos):
            key = (strategy, model)
            if key not in aggregated:
                continue

            data_points = aggregated[key]
            support_vals = [dp[0] for dp in data_points]
            r2_vals = [dp[1] for dp in data_points]

            marker = markers[combo_idx % len(markers)]
            color = combo_to_color[key]

            label = f"{model} - {strategy} - {file_label}"

            ax.plot(
                support_vals,
                r2_vals,
                marker=marker,
                linestyle=line_style,
                linewidth=2,
                markersize=6,
                color=color,
                label=label,
                alpha=0.8,
            )

    # Configure axes
    ax.set_xlabel("Support Set Size", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_xscale("log")
    if support_sizes:
        ax.set_xticks(support_sizes)
        ax.set_xticklabels([str(s) for s in support_sizes], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_title("TTA Performance Comparison", fontsize=14, fontweight="bold")
    
    # Place legend outside plot area
    ax.legend(
        title="Model - Strategy - Run",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    # Add session information as text below the plot
    if all_sessions:
        session_text = "TTA Sessions:\n" + "\n".join(all_sessions)
        fig.text(
            0.02,
            0.02,
            session_text,
            fontsize=8,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        base_name = file_paths[0].stem
        output_path = file_paths[0].parent / f"{base_name}_combined_plot.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot TTA results from log and/or JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="TTA sweep log/JSON file(s) to plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path (if not provided, auto-generated)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )

    args = parser.parse_args()

    # Verify files exist
    for file_path in args.files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

    plot_results(args.files, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
