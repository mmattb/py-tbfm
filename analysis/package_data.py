#!/usr/bin/env python3
"""
Package analysis results into a portable CSV for sharing.

Reads:  results/cv_analysis/full_tta_results_with_baseline.csv
Writes: tbfm_cv_data.csv
"""

import pandas as pd
from pathlib import Path

RESULTS_CSV = Path("results/cv_analysis/full_tta_results_with_baseline.csv")
OUTPUT_CSV  = Path("tbfm_cv_data.csv")


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Not found: {RESULTS_CSV}\nRun analyze_cv_results.py first.")

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    # Drop failed runs
    if "status" in df.columns:
        n_before = len(df)
        df = df[df["status"] == "SUCCESS"].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  Dropped {n_dropped} failed rows")

    # Select and rename essential columns
    cols = {
        "session_id":             "session_id",
        "monkey":                 "monkey",
        "area":                   "area",
        "num_channels":           "num_channels",
        "fold":                   "fold",
        "support_samples":        "support_samples",
        "tta_r2":                 "tta_r2",
        "vanilla_train_r2":       "vanilla_train_r2",
        "vanilla_test_r2":        "vanilla_test_r2",
        "improvement_vs_vanilla": "improvement_vs_vanilla",
    }
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Source CSV missing columns: {missing}")

    clean = df[list(cols.keys())].rename(columns=cols).copy()
    clean = clean.sort_values(
        ["support_samples", "session_id", "fold"]
    ).reset_index(drop=True)

    clean.to_csv(OUTPUT_CSV, index=False)

    # Report
    print(f"\nWrote {len(clean)} rows → {OUTPUT_CSV}")
    print(f"  Sessions:      {clean['session_id'].nunique()}")
    print(f"  Support sizes: {sorted(clean['support_samples'].unique())}")
    print(f"  Folds:         {clean['fold'].nunique()}")
    print(f"  Monkeys:       {sorted(clean['monkey'].unique())}")
    print(f"  Areas:         {sorted(clean['area'].unique())}")
    print(f"\nColumn dtypes:")
    print(clean.dtypes.to_string())


if __name__ == "__main__":
    main()
