#!/usr/bin/env python3
"""
Package recon TTA results into a portable CSV.

Reads:  random_folds_20260101_211200/recon/tta_results_*/fold*/tta_support_*_per_session.csv
        tbfm_cv_data.csv  (vanilla R² baseline and session metadata)
Writes: tbfm_cv_data_recon.csv

The vanilla R² baseline is shared with MAML — it is the session-level performance
of a vanilla model and does not depend on the adaptation strategy.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR   = Path("/home/danmuir/GitHub/py-tbfm")
RECON_DIR  = BASE_DIR / "random_folds_20260101_211200" / "recon"
MAML_CSV   = BASE_DIR / "tbfm_cv_data.csv"
OUTPUT_CSV = BASE_DIR / "tbfm_cv_data_recon.csv"

TTA_DIRS = {
    500:  RECON_DIR / "tta_results_500_20260331_185755",
    1000: RECON_DIR / "tta_results_1000_20260402_061010",
    2500: RECON_DIR / "tta_results_2500_recon_20260317_131715",
    5000: RECON_DIR / "tta_results_5000_20260406_004812",
}


def main():
    parser = argparse.ArgumentParser(description="Package recon TTA data")
    parser.add_argument("--maml-csv", default=str(MAML_CSV),
                        help="Path to MAML tbfm_cv_data.csv for vanilla R² baseline")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path")
    args = parser.parse_args()

    maml_path = Path(args.maml_csv)
    out_path  = Path(args.output)

    if not maml_path.exists():
        sys.exit(f"ERROR: MAML CSV not found: {maml_path}")
    maml = pd.read_csv(maml_path)

    vanilla_lookup = (
        maml.groupby("session_id")
        .agg(
            monkey=("monkey", "first"),
            area=("area", "first"),
            num_channels=("num_channels", "first"),
            vanilla_train_r2=("vanilla_train_r2", "first"),
            vanilla_test_r2=("vanilla_test_r2", "first"),
        )
        .reset_index()
    )

    frames = []
    for support_size, tta_dir in TTA_DIRS.items():
        csvs = sorted(tta_dir.glob("fold*/tta_support_*_per_session.csv"))
        if not csvs:
            print(f"  WARNING: no per_session CSVs found in {tta_dir}, skipping")
            continue
        for csv in csvs:
            df = pd.read_csv(csv)
            df["support_size"] = support_size
            frames.append(df)
        print(f"  support={support_size}: {len(csvs)} fold CSVs from {tta_dir.name}")

    if not frames:
        sys.exit("ERROR: no per-session CSVs found")

    tta = pd.concat(frames, ignore_index=True)

    tta["fold"] = tta["model"].str.extract(r"fold(\d+)").astype(int)
    tta = tta.rename(columns={"session_r2": "tta_r2", "support_size": "support_samples"})

    merged = tta.merge(vanilla_lookup, on="session_id", how="left")

    n_missing = merged["vanilla_test_r2"].isna().sum()
    if n_missing:
        missing = merged.loc[merged["vanilla_test_r2"].isna(), "session_id"].unique()
        print(f"  WARNING: {n_missing} rows missing vanilla R² for sessions: {missing}")

    merged["improvement_vs_vanilla"] = merged["tta_r2"] - merged["vanilla_test_r2"]

    out = merged[[
        "session_id", "monkey", "area", "num_channels",
        "fold", "support_samples",
        "tta_r2", "vanilla_train_r2", "vanilla_test_r2",
        "improvement_vs_vanilla",
    ]].sort_values(["support_samples", "session_id", "fold"]).reset_index(drop=True)

    out.to_csv(out_path, index=False)

    print(f"\nWrote {len(out)} rows → {out_path}")
    print(f"  Sessions:      {out['session_id'].nunique()}")
    print(f"  Support sizes: {sorted(out['support_samples'].unique())}")
    print(f"  Folds:         {out['fold'].nunique()}")
    print(f"  Monkeys:       {sorted(out['monkey'].unique())}")
    print(f"  Areas:         {sorted(out['area'].unique())}")


if __name__ == "__main__":
    main()
