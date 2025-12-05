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

    # Parse vanilla TBFM baseline section
    vanilla_section = re.search(r"Vanilla TBFM Baseline:(.*?)(?=\n\n|\nFresh TBFM|\Z)", content, re.DOTALL)
    if vanilla_section:
        vanilla_pattern = r"Support=\s*(\d+)\s+\|\s+R²=([+-]?\d+\.\d+)"
        for match in re.finditer(vanilla_pattern, vanilla_section.group(1)):
            support_size = int(match.group(1))
            r2 = float(match.group(2))
            results["runs"].append({
                "strategy": "vanilla_tbfm",
                "support_size": support_size,
                "model": "vanilla_tbfm",
                "r2": r2,
            })

    # Parse fresh TBFM baseline section
    fresh_section = re.search(r"Fresh TBFM \(no multisession\) Baseline:(.*?)(?=\n\n|\Z)", content, re.DOTALL)
    if fresh_section:
        fresh_pattern = r"Support=\s*(\d+)\s+\|\s+R²=([+-]?\d+\.\d+)"
        for match in re.finditer(fresh_pattern, fresh_section.group(1)):
            support_size = int(match.group(1))
            r2 = float(match.group(2))
            results["runs"].append({
                "strategy": "fresh_tbfm",
                "support_size": support_size,
                "model": "fresh_tbfm",
                "r2": r2,
            })

    return results


def average_duplicate_runs(runs: List[Dict]) -> List[Dict]:
    """
    Average R² values for runs with the same (strategy, model, support_size).

    This handles cases where multiple training sessions/batches exist for the
    same configuration in a single JSON file.

    Args:
        runs: List of run dictionaries

    Returns:
        List of run dictionaries with duplicates averaged
    """
    # Group runs by (strategy, model, support_size)
    grouped = {}

    for run in runs:
        key = (run["strategy"], run["model"], run["support_size"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(run["r2"])

    # Create new runs list with averaged values
    averaged_runs = []
    for (strategy, model, support_size), r2_values in grouped.items():
        averaged_runs.append({
            "strategy": strategy,
            "model": model,
            "support_size": support_size,
            "r2": np.mean(r2_values),
        })

    return averaged_runs


def parse_json_file(json_path: Path, combine_duplicates: bool = True) -> Dict:
    """
    Parse a TTA sweep JSON file and extract results.

    Args:
        json_path: Path to the JSON file
        combine_duplicates: If True, average duplicate runs with same (model, strategy, support_size)

    Returns:
        Dictionary with same structure as parse_log_file for consistency.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Optionally average duplicate runs (same model, strategy, support_size)
    runs = data.get("runs", [])
    if combine_duplicates:
        runs = average_duplicate_runs(runs)

    results = {
        "metadata": {
            "support_sizes": data["metadata"].get("support_sizes", []),
            "adapt_sessions": data["metadata"].get("adapt_session_ids",
                                                   data["metadata"].get("adapt_sessions", [])),
            "timestamp": data["metadata"].get("timestamp", json_path.stem),
            "source": "json",
        },
        "runs": runs,
    }

    return results


def load_result_file(file_path: Path, combine_duplicates: bool = True) -> Dict:
    """
    Load results from either a log file or JSON file.
    
    Args:
        file_path: Path to log or JSON file
        combine_duplicates: If True, average duplicate runs with same (model, strategy, support_size)
        
    Returns:
        Parsed results dictionary
    """
    if file_path.suffix == ".json":
        return parse_json_file(file_path, combine_duplicates=combine_duplicates)
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


def parse_batch_size(model: str) -> Optional[int]:
    """
    Extract batch size from model name if present.

    Args:
        model: Model name (e.g., "bs500", "bs625", "bs750")

    Returns:
        Batch size as integer, or None if not a batch size model
    """
    if model.startswith("bs") and model[2:].isdigit():
        return int(model[2:])
    return None


def parse_model_name(model: str) -> dict:
    """
    Parse model name to extract components.

    Expected format: {num_bases}_{num_pretrain_sessions}_{training_strategy}
    Example: "50_25_coadapt" -> bases=50, pretrain_sessions=25, training_strategy="coadapt"

    Returns dict with: num_bases, pretrain_sessions, training_strategy, is_shuffle
    """
    if model in ["vanilla_tbfm", "fresh_tbfm"]:
        return {"num_bases": None, "pretrain_sessions": None, "training_strategy": model, "is_shuffle": False}

    parts = model.split("_")
    result = {
        "num_bases": int(parts[0]) if parts[0].isdigit() else None,
        "pretrain_sessions": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
        "training_strategy": None,
        "is_shuffle": "shuffle" in model.lower()
    }

    # Find training strategy (last part that's a word)
    if "coadapt" in model:
        result["training_strategy"] = "coadapt"
    elif "inner" in model:
        result["training_strategy"] = "inner"

    return result


def get_display_name(value: str, dimension: str) -> str:
    """
    Get the display name for a value based on the dimension.

    Args:
        value: The raw value (e.g., "inner", "maml", "coadapt")
        dimension: The dimension type ("training_strategy" or "tta_strategy")

    Returns:
        The display name for plots
    """
    if dimension == "training_strategy":
        mapping = {
            "inner": "MAML Trained",
            "coadapt": "Coadapt Trained",
            "vanilla_tbfm": "Vanilla TBFM",
            "fresh_tbfm": "Fresh TBFM"
        }
        return mapping.get(value, value)
    elif dimension == "tta_strategy":
        mapping = {
            "maml": "MAML TTA",
            "coadapt": "Coadapt TTA",
            "vanilla": "Vanilla TBFM",
            "vanilla_tbfm": "Vanilla TBFM",
            "fresh": "Fresh TBFM",
            "fresh_tbfm": "Fresh TBFM"
        }
        # Use lowercase comparison for case-insensitive matching
        return mapping.get(value.lower() if value else value, value)
    else:
        return value


def get_model_display_name(model: str) -> str:
    """
    Get a human-readable display name for a model.

    Args:
        model: The model name (e.g., "50_25_coadapt", "vanilla_tbfm")

    Returns:
        A readable display name
    """
    # Handle baseline models
    if model in ["vanilla_tbfm", "fresh_tbfm"]:
        return "Vanilla TBFM" if model == "vanilla_tbfm" else "Fresh TBFM"

    # Parse pretrained models
    model_info = parse_model_name(model)

    parts = []
    if model_info["num_bases"] is not None:
        parts.append(f"{model_info['num_bases']} bases")
    if model_info["pretrain_sessions"] is not None:
        parts.append(f"{model_info['pretrain_sessions']} sessions")

    if model_info["training_strategy"]:
        train_display = get_display_name(model_info["training_strategy"], "training_strategy")
        parts.append(f"({train_display})")

    return ", ".join(parts) if parts else model


def aggregate_by_batch_size(
    all_results: List[Tuple[Path, Dict]],
    strategy: Optional[str] = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Aggregate results by batch size.

    Args:
        all_results: List of (file_path, results) tuples
        strategy: Optional TTA strategy to filter by (e.g., "maml")

    Returns:
        Dict mapping batch_size to list of (support_size, r2) tuples
    """
    aggregated = {}

    for file_path, results in all_results:
        for run in results["runs"]:
            model = run["model"]
            run_strategy = run["strategy"]
            support_size = run["support_size"]
            r2 = run["r2"]

            # Filter by strategy if specified
            if strategy and run_strategy.lower() != strategy.lower():
                continue

            # Extract batch size from model name
            batch_size = parse_batch_size(model)
            if batch_size is None:
                continue

            if batch_size not in aggregated:
                aggregated[batch_size] = {}

            if support_size not in aggregated[batch_size]:
                aggregated[batch_size][support_size] = []

            aggregated[batch_size][support_size].append(r2)

    # Average values for each (batch_size, support_size)
    result = {}
    for batch_size, support_dict in aggregated.items():
        result[batch_size] = []
        for support_size in sorted(support_dict.keys()):
            avg_r2 = np.mean(support_dict[support_size])
            result[batch_size].append((support_size, avg_r2))

    return result


def aggregate_by_dimension(
    all_results: List[Tuple[Path, Dict]],
    dimension: str
) -> Dict:
    """
    Aggregate results by a specific dimension (training_strategy, tta_strategy, or pretrain_sessions).

    Args:
        all_results: List of (file_path, results) tuples
        dimension: One of "training_strategy", "tta_strategy", "pretrain_sessions"

    Returns:
        Dict mapping dimension value to list of (support_size, r2) tuples
    """
    aggregated = {}

    for file_path, results in all_results:
        for run in results["runs"]:
            model = run["model"]
            strategy = run["strategy"]
            support_size = run["support_size"]
            r2 = run["r2"]

            # Determine the dimension value
            if dimension == "tta_strategy":
                dim_value = strategy
            elif dimension == "training_strategy":
                model_info = parse_model_name(model)
                dim_value = model_info["training_strategy"]
                if dim_value is None:
                    continue
            elif dimension == "pretrain_sessions":
                model_info = parse_model_name(model)
                dim_value = model_info["pretrain_sessions"]
                if dim_value is None:
                    continue
            else:
                continue

            if dim_value not in aggregated:
                aggregated[dim_value] = {}

            if support_size not in aggregated[dim_value]:
                aggregated[dim_value][support_size] = []

            aggregated[dim_value][support_size].append(r2)

    # Average values for each (dimension_value, support_size)
    result = {}
    for dim_value, support_dict in aggregated.items():
        result[dim_value] = []
        for support_size in sorted(support_dict.keys()):
            avg_r2 = np.mean(support_dict[support_size])
            result[dim_value].append((support_size, avg_r2))

    return result


def plot_results(
    file_paths: List[Path],
    output_path: Optional[Path] = None,
    show: bool = False,
    plot_batch_size: bool = False,
    combine_duplicates: bool = True,
):
    """
    Create comparison plots from log and/or JSON files.

    Generates 4 plots:
    1. Main comparison plot (all models and strategies)
    2. Training strategy comparison
    3. TTA strategy comparison
    4. Pretraining sessions comparison
    5. Batch size comparison (if plot_batch_size=True)

    Args:
        file_paths: List of log/JSON file paths to plot
        output_path: Path to save plot. If None, uses timestamp-based name
        show: Whether to display plot interactively
        combine_duplicates: If True, average duplicate runs with same (model, strategy, support_size)
        plot_batch_size: Whether to generate batch size comparison plot
    """
    # Parse all files
    all_results = []
    for file_path in file_paths:
        print(f"Parsing {file_path.name}...")
        results = load_result_file(file_path, combine_duplicates=combine_duplicates)
        all_results.append((file_path, results))

    if not all_results:
        print("No results found")
        return

    # Combine duplicates globally across all files if requested
    if combine_duplicates and len(all_results) > 1:
        print(f"Combining duplicates across {len(all_results)} files...")
        # Collect all runs from all files
        all_runs = []
        all_support_sizes_set = set()
        all_adapt_sessions = []
        for file_path, results in all_results:
            all_runs.extend(results["runs"])
            all_support_sizes_set.update(results["metadata"]["support_sizes"])
            if results["metadata"]["adapt_sessions"]:
                all_adapt_sessions.extend(results["metadata"]["adapt_sessions"])
        
        # Average duplicate runs
        combined_runs = average_duplicate_runs(all_runs)
        
        # Create a single merged results entry
        merged_results = {
            "metadata": {
                "support_sizes": sorted(all_support_sizes_set),
                "adapt_sessions": sorted(set(all_adapt_sessions)),
                "timestamp": "combined",
                "source": "combined"
            },
            "runs": combined_runs
        }
        
        # Replace all_results with single merged entry
        all_results = [(file_paths[0], merged_results)]
        print(f"Combined {len(all_runs)} runs into {len(combined_runs)} unique configurations")
    
    # Collect all unique strategies, models, and support sizes
    all_strategies = set()
    all_models = set()
    all_support_sizes = set()
    session_to_files = {}  # Map session tuple to list of file names

    # When not combining duplicates, add file identifier to model name to make them unique
    if not combine_duplicates:
        enriched_results = []
        for file_idx, (file_path, results) in enumerate(all_results):
            # Use parent directory name as identifier
            parent_dir_name = file_path.parent.name
            enriched_runs = []
            for run in results["runs"]:
                enriched_run = run.copy()
                # Add parent directory name to model name to make it unique
                enriched_run["model"] = f"{run['model']}__{parent_dir_name}"
                enriched_run["_original_model"] = run["model"]
                enriched_run["_parent_dir"] = parent_dir_name
                enriched_runs.append(enriched_run)
            enriched_results.append((file_path, {**results, "runs": enriched_runs}))
        all_results = enriched_results

    for file_path, results in all_results:
        aggregated = aggregate_runs_by_key(results["runs"])

        for strategy, model in aggregated.keys():
            all_strategies.add(strategy)
            all_models.add(model)

        all_support_sizes.update(results["metadata"]["support_sizes"])

        # Collect session info (deduplicate by session set)
        if results["metadata"]["adapt_sessions"]:
            sessions_tuple = tuple(results["metadata"]["adapt_sessions"])
            if sessions_tuple not in session_to_files:
                session_to_files[sessions_tuple] = []
            session_to_files[sessions_tuple].append(file_path.name)

    all_strategies = sorted(all_strategies)
    all_models = sorted(all_models)
    support_sizes = sorted(all_support_sizes)

    # Create figure with extra space for session info
    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)

    # Map strategies to line styles
    strategy_to_linestyle = {}
    line_styles = ["-", "--", "-.", ":"]
    available_styles = list(line_styles)

    # Assign MAML solid line style
    for strategy in all_strategies:
        if strategy.lower() == "maml":
            strategy_to_linestyle[strategy] = "-"

    # Assign vanilla and fresh the same style
    vanilla_fresh_style = available_styles[-1]
    for strategy in all_strategies:
        if strategy.lower() in ["vanilla", "fresh"]:
            strategy_to_linestyle[strategy] = vanilla_fresh_style

    # Assign other strategies their own styles
    style_idx = 1  # Start from index 1 since solid is taken by MAML
    for strategy in sorted(all_strategies):
        if strategy not in strategy_to_linestyle:
            strategy_to_linestyle[strategy] = available_styles[style_idx % len(available_styles)]
            style_idx += 1

    markers = ["o", "s", "^", "v", "D", "p", "*", "h"]

    # Create color mapping by model (same model = same color across strategies)
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    model_to_color = {model: model_colors[i] for i, model in enumerate(all_models)}

    # Create a mapping for strategy+model combinations
    strategy_model_combos = []
    for strategy in all_strategies:
        for model in all_models:
            strategy_model_combos.append((strategy, model))

    # Plot each file's data
    for file_idx, (file_path, results) in enumerate(all_results):
        aggregated = aggregate_runs_by_key(results["runs"])
        timestamp = results["metadata"]["timestamp"]

        # Plot each strategy+model combination
        for combo_idx, (strategy, model) in enumerate(strategy_model_combos):
            key = (strategy, model)
            if key not in aggregated:
                continue

            data_points = aggregated[key]
            support_vals = [dp[0] for dp in data_points]
            r2_vals = [dp[1] for dp in data_points]

            # Extract original model name if available (for --no-combine-duplicates)
            display_model = model
            file_suffix = ""
            if "__" in model:
                # Extract original model name and parent directory name
                parts = model.rsplit("__", 1)
                if len(parts) == 2:
                    display_model = parts[0]
                    file_suffix = f" ({parts[1]})"
            
            marker = markers[combo_idx % len(markers)]
            color = model_to_color[model]
            line_style = strategy_to_linestyle[strategy]

            model_display = get_model_display_name(display_model)
            strategy_display = get_display_name(strategy, "tta_strategy")

            # Special handling for baseline architectures on composite graph
            if display_model == "vanilla_tbfm" and strategy.lower() in ["vanilla", "vanilla_tbfm"]:
                strategy_display = "Vanilla Architecture"
            elif display_model == "fresh_tbfm" and strategy.lower() in ["fresh", "fresh_tbfm"]:
                strategy_display = "Multisession Architecture"

            label = f"{model_display} - {strategy_display}{file_suffix}"

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

    # Print number of lines plotted for debugging
    num_lines = len(ax.get_lines())
    print(f"Plotted {num_lines} line(s) on main graph")

    # Configure axes
    ax.set_xlabel("Support Set Size", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_xscale("log")
    if support_sizes:
        ax.set_xticks(support_sizes)
        ax.set_xticklabels([str(s) for s in support_sizes], rotation=45)
    
    # Set y-axis limits to show all data properly
    # Get current y limits to find max R² value
    y_data = []
    for line in ax.get_lines():
        y_data.extend(line.get_ydata())
    
    if y_data:
        max_r2 = max(y_data)
        min_r2 = min(y_data)
        # Add some padding
        y_range = max_r2 - min_r2
        padding = max(0.05, y_range * 0.1)  # At least 0.05 or 10% of range
        ax.set_ylim(bottom=min(min_r2 - padding, -0.5), top=max_r2 + padding)
    else:
        ax.set_ylim(bottom=-0.5)
    
    ax.grid(True, alpha=0.3)
    ax.set_title("TTA Performance Comparison", fontsize=14, fontweight="bold")
    
    # Place legend outside plot area
    ax.legend(
        title="Model - Strategy",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    # Add session information on the side below legend
    if session_to_files:
        session_lines = []
        for sessions_tuple, file_names in session_to_files.items():
            if len(file_names) > 1:
                files_str = "\n".join(file_names)
                session_lines.append(f"Files: \n{files_str}")
            else:
                session_lines.append(f"File: {file_names[0]}")
            # Display each session on its own line for narrower box
            session_lines.append("Sessions:")
            for session in sessions_tuple:
                session_lines.append(f"  • {session}")
            session_lines.append("")  # Add blank line between groups

        session_text = "\n".join(session_lines).rstrip()
        fig.text(
            1.05,
            0.5,
            session_text,
            fontsize=8,
            verticalalignment="center",
            transform=ax.transAxes,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        base_name = file_paths[0].stem
        output_path = file_paths[0].parent / f"{base_name}_combined_plot.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    print(f"Main plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Generate comparison plots
    base_path = output_path.parent / output_path.stem

    # 1. Training strategy comparison
    print("\nGenerating training strategy comparison...")
    training_agg = aggregate_by_dimension(all_results, "training_strategy")
    if training_agg:
        fig_train, ax_train = plt.subplots(figsize=(10, 6))
        for train_strat, data_points in sorted(training_agg.items()):
            if train_strat:
                support_vals = [dp[0] for dp in data_points]
                r2_vals = [dp[1] for dp in data_points]
                display_name = get_display_name(train_strat, "training_strategy")
                ax_train.plot(support_vals, r2_vals, marker='o', linewidth=2,
                            markersize=8, label=display_name, alpha=0.8)

        ax_train.set_xlabel("Support Set Size", fontsize=12)
        ax_train.set_ylabel("Test R² (averaged)", fontsize=12)
        ax_train.set_xscale("log")
        ax_train.set_xticks(support_sizes)
        ax_train.set_xticklabels([str(s) for s in support_sizes], rotation=45)
        ax_train.set_ylim(bottom=-0.5)
        ax_train.grid(True, alpha=0.3)
        ax_train.set_title("Training Strategy Comparison", fontsize=14, fontweight="bold")
        ax_train.legend(title="Training Strategy", fontsize=10)
        plt.tight_layout()

        train_path = Path(str(base_path) + "_training_strategy.png")
        fig_train.savefig(train_path, dpi=200, bbox_inches="tight")
        print(f"Training strategy plot saved to: {train_path}")
        if show:
            plt.show()
        else:
            plt.close(fig_train)

    # 2. TTA strategy comparison
    print("\nGenerating TTA strategy comparison...")
    tta_agg = aggregate_by_dimension(all_results, "tta_strategy")
    if tta_agg:
        fig_tta, ax_tta = plt.subplots(figsize=(10, 6))
        for tta_strat, data_points in sorted(tta_agg.items()):
            support_vals = [dp[0] for dp in data_points]
            r2_vals = [dp[1] for dp in data_points]
            display_name = get_display_name(tta_strat, "tta_strategy")
            ax_tta.plot(support_vals, r2_vals, marker='s', linewidth=2,
                       markersize=8, label=display_name, alpha=0.8)

        ax_tta.set_xlabel("Support Set Size", fontsize=12)
        ax_tta.set_ylabel("Test R² (averaged)", fontsize=12)
        ax_tta.set_xscale("log")
        ax_tta.set_xticks(support_sizes)
        ax_tta.set_xticklabels([str(s) for s in support_sizes], rotation=45)
        ax_tta.set_ylim(bottom=-0.5)
        ax_tta.grid(True, alpha=0.3)
        ax_tta.set_title("TTA Strategy Comparison", fontsize=14, fontweight="bold")
        ax_tta.legend(title="TTA Strategy", fontsize=10)
        plt.tight_layout()

        tta_path = Path(str(base_path) + "_tta_strategy.png")
        fig_tta.savefig(tta_path, dpi=200, bbox_inches="tight")
        print(f"TTA strategy plot saved to: {tta_path}")
        if show:
            plt.show()
        else:
            plt.close(fig_tta)

    # 3. Pretraining sessions comparison
    print("\nGenerating pretraining sessions comparison...")
    pretrain_agg = aggregate_by_dimension(all_results, "pretrain_sessions")
    if pretrain_agg:
        fig_pretrain, ax_pretrain = plt.subplots(figsize=(10, 6))
        for num_sessions, data_points in sorted(pretrain_agg.items()):
            support_vals = [dp[0] for dp in data_points]
            r2_vals = [dp[1] for dp in data_points]
            ax_pretrain.plot(support_vals, r2_vals, marker='^', linewidth=2,
                           markersize=8, label=f"{num_sessions} sessions", alpha=0.8)

        ax_pretrain.set_xlabel("Support Set Size", fontsize=12)
        ax_pretrain.set_ylabel("Test R² (averaged)", fontsize=12)
        ax_pretrain.set_xscale("log")
        ax_pretrain.set_xticks(support_sizes)
        ax_pretrain.set_xticklabels([str(s) for s in support_sizes], rotation=45)
        ax_pretrain.set_ylim(bottom=-0.5)
        ax_pretrain.grid(True, alpha=0.3)
        ax_pretrain.set_title("Pretraining Sessions Comparison", fontsize=14, fontweight="bold")
        ax_pretrain.legend(title="Pretraining Sessions", fontsize=10)
        plt.tight_layout()

        pretrain_path = Path(str(base_path) + "_pretrain_sessions.png")
        fig_pretrain.savefig(pretrain_path, dpi=200, bbox_inches="tight")
        print(f"Pretraining sessions plot saved to: {pretrain_path}")
        if show:
            plt.show()
        else:
            plt.close(fig_pretrain)

    # 4. Pretrain size vs R² at specific support size
    print("\nGenerating pretrain size vs R² comparison...")
    generate_pretrain_size_plot(all_results, base_path, show=show)

    # 4. Batch size comparison (if enabled)
    if plot_batch_size:
        print("\nGenerating batch size comparison...")
        
        # Get all unique strategies from batch size models
        batch_strategies = set()
        for file_path, results in all_results:
            for run in results["runs"]:
                if parse_batch_size(run["model"]) is not None:
                    batch_strategies.add(run["strategy"])
        
        if batch_strategies:
            # Create a subplot for each strategy
            n_strategies = len(batch_strategies)
            fig_batch, axes = plt.subplots(1, n_strategies, figsize=(7*n_strategies, 6), squeeze=False)
            axes = axes.flatten()
            
            for idx, strategy in enumerate(sorted(batch_strategies)):
                ax_batch = axes[idx]
                batch_agg = aggregate_by_batch_size(all_results, strategy=strategy)
                
                if batch_agg:
                    colors = plt.cm.viridis(np.linspace(0, 0.9, len(batch_agg)))
                    for color_idx, (batch_size, data_points) in enumerate(sorted(batch_agg.items())):
                        support_vals = [dp[0] for dp in data_points]
                        r2_vals = [dp[1] for dp in data_points]
                        ax_batch.plot(support_vals, r2_vals, marker='o', linewidth=2,
                                    markersize=8, label=f"BS={batch_size}", 
                                    color=colors[color_idx], alpha=0.8)
                    
                    ax_batch.set_xlabel("Support Set Size", fontsize=12)
                    ax_batch.set_ylabel("Test R² (averaged)", fontsize=12)
                    ax_batch.set_xscale("log")
                    if support_sizes:
                        ax_batch.set_xticks(support_sizes)
                        ax_batch.set_xticklabels([str(s) for s in support_sizes], rotation=45)
                    ax_batch.set_ylim(bottom=-0.5)
                    ax_batch.grid(True, alpha=0.3)
                    strategy_display = get_display_name(strategy, "tta_strategy")
                    ax_batch.set_title(f"Batch Size Comparison\n({strategy_display})", 
                                     fontsize=14, fontweight="bold")
                    ax_batch.legend(title="Batch Size", fontsize=10)
            
            plt.tight_layout()
            batch_path = Path(str(base_path) + "_batch_size.png")
            fig_batch.savefig(batch_path, dpi=200, bbox_inches="tight")
            print(f"Batch size plot saved to: {batch_path}")
            if show:
                plt.show()
            else:
                plt.close(fig_batch)
        else:
            print("No batch size data found (models must be named like 'bs500', 'bs625', etc.)")

    print("\nAll comparison plots generated!")

    # Generate summary statistics table
    print("\nGenerating summary statistics table...")
    generate_summary_table(all_results, base_path)

    # Generate slide-ready visual table
    print("\nGenerating slide-ready table...")
    generate_slide_table(all_results, base_path)


def generate_pretrain_size_plot(
    all_results: List[Tuple[Path, Dict]], 
    base_path: Path,
    support_size: int = 1000,
    show: bool = False
):
    """
    Generate a plot of pretrain session count vs R² for a specific support size.
    
    Creates separate lines for each combination of training strategy and TTA strategy.
    
    Args:
        all_results: List of (file_path, results) tuples
        base_path: Base path for saving the plot
        support_size: Support size to filter for (default: 1000)
        show: Whether to display the plot
    """
    import pandas as pd
    
    # Collect all data with parsed model info
    data = []
    for file_path, results in all_results:
        for run in results["runs"]:
            model_info = parse_model_name(run["model"])
            data.append({
                "Model": run["model"],
                "TTA Strategy": run["strategy"],
                "Support Size": run["support_size"],
                "R²": run["r2"],
                "Training Strategy": model_info["training_strategy"],
                "Pretrain Sessions": model_info["pretrain_sessions"],
                "Is Shuffle": model_info["is_shuffle"],
            })
    
    df = pd.DataFrame(data)
    
    # Filter for the specific support size
    df_filtered = df[df["Support Size"] == support_size].copy()
    
    # Remove baseline models
    df_filtered = df_filtered[~df_filtered["TTA Strategy"].isin(["vanilla_tbfm", "fresh_tbfm"])].copy()
    
    # Remove rows with missing pretrain sessions
    df_filtered = df_filtered[df_filtered["Pretrain Sessions"].notna()].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No data found for support size {support_size}")
        return
    
    # Group by training strategy, TTA strategy, pretrain sessions, AND shuffle status, then average R²
    grouped = df_filtered.groupby(
        ["Training Strategy", "TTA Strategy", "Pretrain Sessions", "Is Shuffle"]
    )["R²"].mean().reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot each combination of training strategy and TTA strategy
    combo_idx = 0
    for (train_strat, tta_strat), group in grouped.groupby(["Training Strategy", "TTA Strategy"]):
        if train_strat is None:
            continue
        
        train_display = get_display_name(train_strat, "training_strategy")
        tta_display = get_display_name(tta_strat, "tta_strategy")
        
        # Separate shuffle and non-shuffle
        group_non_shuffle = group[group["Is Shuffle"] == False].sort_values("Pretrain Sessions")
        group_shuffle = group[group["Is Shuffle"] == True].sort_values("Pretrain Sessions")
        
        marker = markers[combo_idx % len(markers)]
        color = colors[combo_idx % len(colors)]
        
        # Plot non-shuffle models
        if len(group_non_shuffle) > 0:
            pretrain_sessions = group_non_shuffle["Pretrain Sessions"].values
            r2_values = group_non_shuffle["R²"].values
            label = f"{train_display} + {tta_display}"
            ax.plot(pretrain_sessions, r2_values, marker=marker, linestyle='-',
                    linewidth=2, markersize=10, label=label, color=color, alpha=0.8)
        
        # Plot shuffle models with lighter color and different marker
        if len(group_shuffle) > 0:
            pretrain_sessions = group_shuffle["Pretrain Sessions"].values
            r2_values = group_shuffle["R²"].values
            label = f"{train_display} + {tta_display} (shuffle)"
            shuffle_marker = '*' if marker != '*' else 'P'
            ax.plot(pretrain_sessions, r2_values, marker=shuffle_marker, linestyle='--',
                    linewidth=2, markersize=12, label=label, color=color, alpha=0.5)
        
        combo_idx += 1
    
    ax.set_xlabel('Number of Pretraining Sessions', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title(f'Pretraining Sessions vs R² (Support Size = {support_size})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    # Set x-axis to show only the actual pretrain session counts
    pretrain_counts = sorted(df_filtered["Pretrain Sessions"].unique())
    ax.set_xticks(pretrain_counts)
    ax.set_xticklabels([str(int(c)) for c in pretrain_counts])
    
    plt.tight_layout()
    
    # Save plot
    pretrain_size_path = Path(str(base_path) + f"_pretrain_size_vs_r2_support{support_size}.png")
    fig.savefig(pretrain_size_path, dpi=200, bbox_inches='tight')
    print(f"Pretrain size vs R² plot saved to: {pretrain_size_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_slide_table(all_results: List[Tuple[Path, Dict]], base_path: Path):
    """
    Generate a visual table as an image for presentation slides.

    Creates a table showing relative improvements from different factors:
    - MAML vs coadapt TTA
    - MAML vs coadapt training
    - 25 vs 5 pretrain sessions
    - Ideal (best pretrained) vs baseline

    Args:
        all_results: List of (file_path, results) tuples
        base_path: Base path for saving the table image
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors

    # Collect all data with parsed model info
    data = []
    for file_path, results in all_results:
        for run in results["runs"]:
            model_info = parse_model_name(run["model"])
            data.append({
                "Model": run["model"],
                "TTA Strategy": run["strategy"],
                "Training Strategy": model_info["training_strategy"],
                "Pretrain Sessions": model_info["pretrain_sessions"],
                "Support Size": run["support_size"],
                "R²": run["r2"]
            })

    df = pd.DataFrame(data)

    # Select key support sizes to show sample efficiency
    key_sizes = [100, 250, 500, 1000, 5000]
    df_filtered = df[df["Support Size"].isin(key_sizes)].copy()

    # Remove baseline models for comparisons (we only compare pretrained models)
    df_pretrained = df_filtered[~df_filtered["TTA Strategy"].isin(["vanilla_tbfm", "fresh_tbfm"])].copy()

    # Calculate improvements for each factor
    comparisons = []

    # 1. MAML vs coadapt TTA (holding other factors constant)
    # Compare models with same training strategy and pretrain sessions but different TTA
    for train_strat in df_pretrained["Training Strategy"].dropna().unique():
        for sessions in df_pretrained["Pretrain Sessions"].dropna().unique():
            maml_data = df_pretrained[
                (df_pretrained["TTA Strategy"].str.lower() == "maml") &
                (df_pretrained["Training Strategy"] == train_strat) &
                (df_pretrained["Pretrain Sessions"] == sessions)
            ]
            coadapt_data = df_pretrained[
                (df_pretrained["TTA Strategy"].str.lower() == "coadapt") &
                (df_pretrained["Training Strategy"] == train_strat) &
                (df_pretrained["Pretrain Sessions"] == sessions)
            ]

            if len(maml_data) > 0 and len(coadapt_data) > 0:
                for size in key_sizes:
                    maml_r2 = maml_data[maml_data["Support Size"] == size]["R²"].mean()
                    coadapt_r2 = coadapt_data[coadapt_data["Support Size"] == size]["R²"].mean()
                    if not pd.isna(maml_r2) and not pd.isna(coadapt_r2):
                        improvement = maml_r2 - coadapt_r2
                        comparisons.append({
                            "Factor": "MAML vs Co-adapt TTA",
                            "Support Size": size,
                            "Improvement": improvement
                        })

    # 2. Inner vs coadapt training (holding other factors constant)
    for tta_strat in ["maml", "coadapt"]:
        for sessions in df_pretrained["Pretrain Sessions"].dropna().unique():
            inner_data = df_pretrained[
                (df_pretrained["TTA Strategy"].str.lower() == tta_strat) &
                (df_pretrained["Training Strategy"] == "inner") &
                (df_pretrained["Pretrain Sessions"] == sessions)
            ]
            coadapt_train_data = df_pretrained[
                (df_pretrained["TTA Strategy"].str.lower() == tta_strat) &
                (df_pretrained["Training Strategy"] == "coadapt") &
                (df_pretrained["Pretrain Sessions"] == sessions)
            ]

            if len(inner_data) > 0 and len(coadapt_train_data) > 0:
                for size in key_sizes:
                    inner_r2 = inner_data[inner_data["Support Size"] == size]["R²"].mean()
                    coadapt_r2 = coadapt_train_data[coadapt_train_data["Support Size"] == size]["R²"].mean()
                    if not pd.isna(inner_r2) and not pd.isna(coadapt_r2):
                        improvement = inner_r2 - coadapt_r2
                        comparisons.append({
                            "Factor": "MAML vs Co-adapt Training",
                            "Support Size": size,
                            "Improvement": improvement
                        })

    # 3. More pretrain sessions (25 vs 5, holding other factors constant)
    for tta_strat in ["maml", "coadapt"]:
        for train_strat in df_pretrained["Training Strategy"].dropna().unique():
            sessions_25 = df_pretrained[
                (df_pretrained["TTA Strategy"].str.lower() == tta_strat) &
                (df_pretrained["Training Strategy"] == train_strat) &
                (df_pretrained["Pretrain Sessions"] == 25)
            ]
            sessions_5 = df_pretrained[
                (df_pretrained["TTA Strategy"].str.lower() == tta_strat) &
                (df_pretrained["Training Strategy"] == train_strat) &
                (df_pretrained["Pretrain Sessions"] == 5)
            ]

            if len(sessions_25) > 0 and len(sessions_5) > 0:
                for size in key_sizes:
                    r2_25 = sessions_25[sessions_25["Support Size"] == size]["R²"].mean()
                    r2_5 = sessions_5[sessions_5["Support Size"] == size]["R²"].mean()
                    if not pd.isna(r2_25) and not pd.isna(r2_5):
                        improvement = r2_25 - r2_5
                        comparisons.append({
                            "Factor": "25 vs 5 Pretrain Sessions",
                            "Support Size": size,
                            "Improvement": improvement
                        })

    # 4. Ideal (best pretrained) vs Baseline
    # Get baseline performance
    baseline_df = df_filtered[df_filtered["TTA Strategy"].isin(["vanilla_tbfm", "fresh_tbfm"])]

    if len(baseline_df) > 0 and len(df_pretrained) > 0:
        for size in key_sizes:
            baseline_r2 = baseline_df[baseline_df["Support Size"] == size]["R²"].mean()
            pretrained_r2 = df_pretrained[df_pretrained["Support Size"] == size]["R²"].max()

            if not pd.isna(baseline_r2) and not pd.isna(pretrained_r2):
                improvement = pretrained_r2 - baseline_r2
                comparisons.append({
                    "Factor": "Ideal vs Baseline",
                    "Support Size": size,
                    "Improvement": improvement
                })

    if not comparisons:
        print("Warning: No valid comparisons found for table generation")
        return

    comp_df = pd.DataFrame(comparisons)

    # Average improvements across all matching pairs for each factor
    pivot = comp_df.pivot_table(
        values='Improvement',
        index='Factor',
        columns='Support Size',
        aggfunc='mean'
    )

    # Reorder rows
    row_order = ["MAML vs Co-adapt TTA", "MAML vs Co-adapt Training", "25 vs 5 Pretrain Sessions", "Ideal vs Baseline"]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data with + signs for positive values (R² point differences)
    cell_text = []
    for idx, row in pivot.iterrows():
        cell_text.append([f"{val:+.3f}" if not pd.isna(val) else "N/A" for val in row])

    # Create the table
    table = ax.table(
        cellText=cell_text,
        rowLabels=pivot.index.tolist(),
        colLabels=[f"{int(s)}" for s in pivot.columns],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)

    # Color code: green = positive (improvement), red = negative (degradation)
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdYlGn

    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                color = cmap(norm(val))
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.7)
                table[(i+1, j)].set_text_props(weight='bold', size=11)

    # Style headers
    for j in range(len(pivot.columns)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(weight='bold', color='white', size=12)

    # Style row labels
    for i in range(len(pivot)):
        table[(i+1, -1)].set_facecolor('#34495E')
        table[(i+1, -1)].set_text_props(weight='bold', size=10, color='white')

    # Add title and subtitle
    plt.text(0.5, 0.96, 'Relative Improvements in Sample Efficiency',
             transform=fig.transFigure, fontsize=15, fontweight='bold',
             ha='center', va='top')
    plt.text(0.5, 0.92, 'R² point improvement from each factor across support set sizes',
             transform=fig.transFigure, fontsize=10, style='italic',
             ha='center', va='top', color='#555')

    plt.tight_layout()

    # Save the table
    table_path = Path(str(base_path) + "_table.png")
    fig.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Slide table saved to: {table_path}")
    plt.close(fig)


def generate_summary_table(all_results: List[Tuple[Path, Dict]], base_path: Path):
    """
    Generate a summary statistics table for all results.

    Args:
        all_results: List of (file_path, results) tuples
        base_path: Base path for saving the table
    """
    import pandas as pd

    # Collect all data
    data = []
    for file_path, results in all_results:
        for run in results["runs"]:
            data.append({
                "Model": run["model"],
                "Strategy": run["strategy"],
                "Support Size": run["support_size"],
                "R²": run["r2"]
            })

    df = pd.DataFrame(data)

    # Create summary statistics
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("TTA PERFORMANCE SUMMARY STATISTICS")
    summary_lines.append("=" * 100)
    summary_lines.append("")

    # Overall statistics
    summary_lines.append("OVERALL STATISTICS:")
    summary_lines.append(f"  Total runs: {len(df)}")
    summary_lines.append(f"  Mean R²: {df['R²'].mean():.4f}")
    summary_lines.append(f"  Median R²: {df['R²'].median():.4f}")
    summary_lines.append(f"  Std Dev: {df['R²'].std():.4f}")
    summary_lines.append(f"  Min R²: {df['R²'].min():.4f}")
    summary_lines.append(f"  Max R²: {df['R²'].max():.4f}")
    summary_lines.append("")

    # Best performing configurations
    summary_lines.append("TOP 10 CONFIGURATIONS (by R²):")
    top_configs = df.nlargest(10, 'R²')
    for idx, row in top_configs.iterrows():
        summary_lines.append(
            f"  {row['Model']:30s} | {row['Strategy']:20s} | "
            f"Support={row['Support Size']:5d} | R²={row['R²']:7.4f}"
        )
    summary_lines.append("")

    # Performance by model
    summary_lines.append("PERFORMANCE BY MODEL:")
    model_stats = df.groupby('Model')['R²'].agg(['mean', 'std', 'min', 'max', 'count'])
    model_stats = model_stats.sort_values('mean', ascending=False)
    summary_lines.append(f"  {'Model':<30s} | {'Mean R²':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s} | {'Count':>5s}")
    summary_lines.append(f"  {'-'*30} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*5}")
    for model, row in model_stats.iterrows():
        summary_lines.append(
            f"  {model:<30s} | {row['mean']:8.4f} | {row['std']:8.4f} | "
            f"{row['min']:8.4f} | {row['max']:8.4f} | {int(row['count']):5d}"
        )
    summary_lines.append("")

    # Performance by strategy
    summary_lines.append("PERFORMANCE BY TTA STRATEGY:")
    strategy_stats = df.groupby('Strategy')['R²'].agg(['mean', 'std', 'min', 'max', 'count'])
    strategy_stats = strategy_stats.sort_values('mean', ascending=False)
    summary_lines.append(f"  {'Strategy':<20s} | {'Mean R²':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s} | {'Count':>5s}")
    summary_lines.append(f"  {'-'*20} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*5}")
    for strategy, row in strategy_stats.iterrows():
        summary_lines.append(
            f"  {strategy:<20s} | {row['mean']:8.4f} | {row['std']:8.4f} | "
            f"{row['min']:8.4f} | {row['max']:8.4f} | {int(row['count']):5d}"
        )
    summary_lines.append("")

    # Performance by support size
    summary_lines.append("PERFORMANCE BY SUPPORT SIZE:")
    support_stats = df.groupby('Support Size')['R²'].agg(['mean', 'std', 'min', 'max', 'count'])
    support_stats = support_stats.sort_index()
    summary_lines.append(f"  {'Support Size':>12s} | {'Mean R²':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s} | {'Count':>5s}")
    summary_lines.append(f"  {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*5}")
    for support_size, row in support_stats.iterrows():
        summary_lines.append(
            f"  {support_size:12d} | {row['mean']:8.4f} | {row['std']:8.4f} | "
            f"{row['min']:8.4f} | {row['max']:8.4f} | {int(row['count']):5d}"
        )
    summary_lines.append("")

    # Performance by model+strategy combination
    summary_lines.append("PERFORMANCE BY MODEL + STRATEGY:")
    combo_stats = df.groupby(['Model', 'Strategy'])['R²'].agg(['mean', 'std', 'count'])
    combo_stats = combo_stats.sort_values('mean', ascending=False)
    summary_lines.append(f"  {'Model':<30s} | {'Strategy':<20s} | {'Mean R²':>8s} | {'Std':>8s} | {'Count':>5s}")
    summary_lines.append(f"  {'-'*30} | {'-'*20} | {'-'*8} | {'-'*8} | {'-'*5}")
    for (model, strategy), row in combo_stats.iterrows():
        summary_lines.append(
            f"  {model:<30s} | {strategy:<20s} | {row['mean']:8.4f} | "
            f"{row['std']:8.4f} | {int(row['count']):5d}"
        )
    summary_lines.append("")

    summary_lines.append("=" * 100)

    # Save to file
    summary_text = "\n".join(summary_lines)
    summary_path = Path(str(base_path) + "_summary_stats.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    print(f"Summary statistics saved to: {summary_path}")

    # Also print to console
    print("\n" + summary_text)

    # Save detailed CSV
    csv_path = Path(str(base_path) + "_detailed_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Detailed results CSV saved to: {csv_path}")


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
    parser.add_argument(
        "--no-combine-duplicates",
        action="store_true",
        help="Don't combine/average runs with same model, strategy, and support size",
    )
    parser.add_argument(
        "--batch-size",
        action="store_true",
        help="Generate batch size comparison plot (models must be named like 'bs500', 'bs625', etc.)",
    )

    args = parser.parse_args()

    # Verify files exist
    for file_path in args.files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

    plot_results(
        args.files,
        output_path=args.output,
        show=args.show,
        combine_duplicates=not args.no_combine_duplicates,
        plot_batch_size=args.batch_size)


if __name__ == "__main__":
    main()
