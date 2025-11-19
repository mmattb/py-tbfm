#!/usr/bin/env python3
"""
Plot TTA sweep results from log files.

This script reads TTA sweep logs and generates comparison plots.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: Path) -> Dict:
    """
    Parse a TTA sweep log file and extract results.

    Returns:
        Dictionary with structure:
        {
            'metadata': {'support_sizes': [...], 'adapt_sessions': [...]},
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
    # Format: "Run complete | Strategy=XXX | Support=YYY | Model=ZZZ | R²=RRR"
    # Also handle "Complete" as an alternative to "Run complete"
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
    log_paths: List[Path],
    output_path: Path | None = None,
    show: bool = False,
):
    """
    Create comparison plots from log files.

    Args:
        log_paths: List of log file paths to plot
        output_path: Path to save plot. If None, uses timestamp-based name
        show: Whether to display plot interactively
    """
    # Parse all logs
    all_results = []
    for log_path in log_paths:
        print(f"Parsing {log_path.name}...")
        results = parse_log_file(log_path)
        all_results.append(results)

    if not all_results:
        print("No results found in logs")
        return

    # Extract metadata from first log
    metadata = all_results[0]["metadata"]
    support_sizes = metadata["support_sizes"]

    # Collect all unique strategies and models
    all_strategies = set()
    all_models = set()
    all_aggregated = []

    for results in all_results:
        aggregated = aggregate_runs_by_key(results["runs"])
        all_aggregated.append(aggregated)

        for strategy, model in aggregated.keys():
            all_strategies.add(strategy)
            all_models.add(model)

    all_strategies = sorted(all_strategies)
    all_models = sorted(all_models)

    # Create a single figure with all data on one plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Use different line styles for strategies and markers for models
    strategies_list = list(all_strategies)
    models_list = list(all_models)
    
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "p", "*"]

    # Plot each strategy+model combination
    for strategy_idx, strategy in enumerate(strategies_list):
        for model_idx, model in enumerate(models_list):
            points_by_log = []

            # Collect data from all logs for this strategy+model
            for aggregated in all_aggregated:
                key = (strategy, model)
                if key in aggregated:
                    points_by_log.append(aggregated[key])

            # Plot if data exists in any log
            if points_by_log:
                data_points = points_by_log[0]
                support_vals = [dp[0] for dp in data_points]
                r2_vals = [dp[1] for dp in data_points]

                line_style = line_styles[strategy_idx % len(line_styles)]
                marker = markers[model_idx % len(markers)]

                ax.plot(
                    support_vals,
                    r2_vals,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=2,
                    markersize=6,
                    label=f"{model} - {strategy}",
                )

    ax.set_xlabel("Support Set Size", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(support_sizes)
    ax.set_xticklabels([str(s) for s in support_sizes], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_title("TTA Performance vs Support Size", fontsize=14)
    ax.legend(title="Model - Strategy", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    # Determine output path
    if output_path is None:
        # Use first log's name as base
        base_name = log_paths[0].stem
        output_path = log_paths[0].parent / f"{base_name}_plot.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot TTA sweep results from log files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="TTA sweep log file(s) to plot",
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

    # Verify log files exist
    for log_file in args.log_files:
        if not log_file.exists():
            print(f"Error: Log file not found: {log_file}")
            return

    plot_results(args.log_files, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
