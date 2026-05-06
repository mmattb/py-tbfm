#!/usr/bin/env python3
"""
Compile ablation results into a per-session table for Matthew.

Usage:
    python analysis/compile_ablation_table.py \\
        --ablations-dir ablations_20260101_120000 \\
        [--baseline random_folds_20260101_211200/fold0] \\
        [--support-size 500]

Output:
    <ablations_dir>/ablation_table.csv   — per-session rows
    <ablations_dir>/ablation_summary.csv — mean ± std per ablation
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import numpy as np


ABLATION_NAMES = [
    "baseline",
    "zscore_norm",
    "no_tanh",
    "no_ae_recon",
    "with_ortho",
    "no_rest",
    "no_l2",
    "no_adapt_ae",
]

ABLATION_LABELS = {
    "baseline":    "Baseline (fold0 mirror)",
    "zscore_norm": "Z-score normalizer",
    "no_tanh":     "No tanh on basis weights",
    "no_ae_recon": "No AE recon loss (λ=0)",
    "with_ortho":  "With ortho penalty (λ_ortho=0.05)",
    "no_rest":     "No c_rest (zeroed)",
    "no_l2":       "No L2 on stim embeddings (λ_l2=0)",
    "no_adapt_ae": "No AE fine-tune at TTA (PCA only)",
}


def r2_from_tensors(y: "torch.Tensor", y_hat: "torch.Tensor") -> float:
    y = y.float().reshape(-1)
    y_hat = y_hat.float().reshape(-1)
    ss_res = ((y - y_hat) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_train_results(model_dir: Path):
    """Load per-session train/test R² from r.torch."""
    r_path = model_dir / "r.torch"
    if not r_path.exists():
        return None
    r = torch.load(r_path, map_location="cpu")
    # Compute per-session train R² from y / y_hat if not already stored
    if "final_train_r2s" not in r and "y" in r and "y_hat" in r:
        r["final_train_r2s"] = {
            sid: r2_from_tensors(r["y"][sid], r["y_hat"][sid])
            for sid in r["y"]
        }
    return r


def load_held_in_sessions(model_dir: Path):
    hisi_path = model_dir / "hisi.torch"
    if not hisi_path.exists():
        return []
    return torch.load(hisi_path, map_location="cpu")


def load_tta_results(tta_dir: Path, support_size: int):
    """
    Load per-session TTA R² from tta_dir.
    Looks for tta_support_<N>_per_session.csv produced by tta_testing.py.
    Returns (test_r2_dict, train_r2_dict): each {session_id: r2}
    """
    if not tta_dir.exists():
        return {}, {}

    # Glob for tta_support_*_per_session.csv and filter by support_size column
    import glob as _glob
    csvs = sorted(_glob.glob(str(tta_dir / "tta_support_*_per_session.csv")))
    for csv_path in csvs:
        test_results = {}
        train_results = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sz = row.get("support_size", "")
                try:
                    if sz and int(sz) != support_size:
                        continue
                except (ValueError, TypeError):
                    pass
                sid = row.get("session_id", "")
                try:
                    test_results[sid] = float(row.get("session_r2", ""))
                except (ValueError, TypeError):
                    pass
                try:
                    train_results[sid] = float(row.get("session_train_r2", ""))
                except (ValueError, TypeError):
                    pass
        if test_results:
            return test_results, train_results

    # Fallback: try torch results
    for fname in ["results.torch", "r.torch"]:
        r_path = tta_dir / fname
        if r_path.exists():
            try:
                r = torch.load(r_path, map_location="cpu")
                if isinstance(r, dict) and "final_test_r2s" in r:
                    return r["final_test_r2s"], {}
            except Exception:
                pass

    return {}, {}


def animal_from_session(session_id: str) -> str:
    return session_id.split("_")[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Compile ablation results table")
    parser.add_argument("--ablations-dir", type=Path, required=True,
                        help="Directory containing ablation subdirs")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Optional external baseline model dir (e.g. fold0)")
    parser.add_argument("--support-size", type=int, default=2500,
                        help="Support size to report for TTA R²")
    return parser.parse_args()


def main():
    args = parse_args()
    ablations_dir = args.ablations_dir.resolve()
    support_size = args.support_size

    rows = []
    summary = {}

    for abl in ABLATION_NAMES:
        # Model dir (training results)
        if abl == "no_adapt_ae":
            model_dir = ablations_dir / "baseline"
        else:
            model_dir = ablations_dir / abl

        # Use external baseline if provided and this is the baseline row
        if abl == "baseline" and args.baseline is not None:
            model_dir = args.baseline.resolve()

        if not model_dir.exists():
            print(f"  [{abl}] model dir not found: {model_dir}")
            continue

        # TTA results dir — try with support_size suffix first, fall back to legacy
        tta_dir = ablations_dir / f"tta_{abl}_{support_size}"
        if not tta_dir.exists():
            tta_dir = ablations_dir / f"tta_{abl}"

        r = load_train_results(model_dir)
        held_in = set(load_held_in_sessions(model_dir))
        tta_r2_map, tta_train_r2_map = load_tta_results(tta_dir, support_size)

        if r is None:
            print(f"  [{abl}] r.torch not found in {model_dir}")
            continue

        # Per-session train/test R² from r.torch
        per_session_train = r.get("final_train_r2s", {})
        per_session_test = r.get("final_test_r2s", {})

        # Collect all session IDs we know about
        all_sessions = set(per_session_test.keys()) | set(tta_r2_map.keys()) | held_in

        label = ABLATION_LABELS.get(abl, abl)
        abl_tta_r2s = []
        abl_train_r2s = []

        for sid in sorted(all_sessions):
            split = "held-in" if sid in held_in else "held-out (TTA)"
            animal = animal_from_session(sid)
            train_r2 = per_session_train.get(sid)
            test_r2 = per_session_test.get(sid)
            tta_r2 = tta_r2_map.get(sid)
            tta_train_r2 = tta_train_r2_map.get(sid)

            rows.append({
                "ablation": abl,
                "ablation_label": label,
                "session": sid,
                "animal": animal,
                "split": split,
                "train_r2": f"{train_r2:.4f}" if train_r2 is not None else "",
                "test_r2": f"{test_r2:.4f}" if test_r2 is not None else "",
                f"tta_r2_{support_size}": f"{tta_r2:.4f}" if tta_r2 is not None else "",
                f"tta_train_r2_{support_size}": f"{tta_train_r2:.4f}" if tta_train_r2 is not None else "",
            })

            if test_r2 is not None and sid in held_in:
                abl_train_r2s.append(test_r2)
            if tta_r2 is not None and sid not in held_in:
                abl_tta_r2s.append(tta_r2)

        summary[abl] = {
            "label": label,
            "n_held_in": len(abl_train_r2s),
            "held_in_test_r2_mean": np.mean(abl_train_r2s) if abl_train_r2s else float("nan"),
            "held_in_test_r2_std": np.std(abl_train_r2s) if abl_train_r2s else float("nan"),
            "n_tta": len(abl_tta_r2s),
            "tta_r2_mean": np.mean(abl_tta_r2s) if abl_tta_r2s else float("nan"),
            "tta_r2_std": np.std(abl_tta_r2s) if abl_tta_r2s else float("nan"),
        }

    # Write per-session table
    table_path = ablations_dir / "ablation_table.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-session table written to: {table_path}")
    else:
        print("\nNo rows to write.")

    # Write summary
    summary_path = ablations_dir / "ablation_summary.csv"
    summary_fields = ["ablation", "label", "n_held_in", "held_in_test_r2_mean",
                      "held_in_test_r2_std", "n_tta", "tta_r2_mean", "tta_r2_std"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for abl, s in summary.items():
            writer.writerow({"ablation": abl, **s})
    print(f"Summary table written to: {summary_path}")

    # Print console summary
    print(f"\n{'Ablation':<35} {'Held-in test R²':>20} {'TTA R² @{:d}':>20}".format(support_size))
    print("-" * 80)
    for abl, s in summary.items():
        hi = f"{s['held_in_test_r2_mean']:.3f} ± {s['held_in_test_r2_std']:.3f}" if s["n_held_in"] else "—"
        tta = f"{s['tta_r2_mean']:.3f} ± {s['tta_r2_std']:.3f}" if s["n_tta"] else "—"
        print(f"{s['label']:<35} {hi:>20} {tta:>20}")


if __name__ == "__main__":
    main()
