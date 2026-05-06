#!/usr/bin/env python3
"""
Package coadapt TTA results into a portable CSV.

Reads:  random_folds_coadapt_*/tta_results_*/tta_summary.csv  (per-fold TTA R²)
        tbfm_cv_data.csv                                        (vanilla R² baseline)
Writes: tbfm_cv_data_coadapt.csv

The vanilla R² baseline is shared with MAML — it is the session-level performance
of a vanilla model and does not depend on the adaptation strategy.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

COADAPT_DIR  = Path("random_folds_coadapt_20260225_015603")
MAML_CSV     = Path("tbfm_cv_data.csv")
OUTPUT_CSV   = Path("tbfm_cv_data_coadapt.csv")

TTA_DIRS = {
    500:  COADAPT_DIR / "tta_results_500_20260309_125123",
    1000: COADAPT_DIR / "tta_results_1000_20260310_021233",
    2500: COADAPT_DIR / "tta_results_2500_20260310_063929",
    5000: COADAPT_DIR / "tta_results_5000_20260310_151044",
}


def main():
    parser = argparse.ArgumentParser(description="Package coadapt TTA data")
    parser.add_argument("--maml-csv", default=str(MAML_CSV),
                        help="Path to MAML tbfm_cv_data.csv for vanilla R² baseline")
    parser.add_argument("--output", default=str(OUTPUT_CSV),
                        help="Output CSV path")
    args = parser.parse_args()

    maml_path = Path(args.maml_csv)
    out_path  = Path(args.output)

    # ── Vanilla R² lookup from MAML CSV ────────────────────────────────────────
    if not maml_path.exists():
        sys.exit(f"ERROR: MAML CSV not found: {maml_path}")
    maml = pd.read_csv(maml_path)

    # vanilla_r2 is fold-invariant per session; take one value per session
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

    # ── Read coadapt TTA summaries ──────────────────────────────────────────────
    frames = []
    for support_size, tta_dir in TTA_DIRS.items():
        summary_csv = tta_dir / "tta_summary.csv"
        if not summary_csv.exists():
            print(f"  WARNING: not found, skipping {summary_csv}")
            continue
        df = pd.read_csv(summary_csv)
        if "status" in df.columns:
            n_before = len(df)
            df = df[df["status"] == "SUCCESS"].copy()
            n_dropped = n_before - len(df)
            if n_dropped:
                print(f"  support={support_size}: dropped {n_dropped} failed rows")
        frames.append(df)
        print(f"  support={support_size}: {len(df)} rows from {summary_csv}")

    if not frames:
        sys.exit("ERROR: no TTA summary CSVs found")

    tta = pd.concat(frames, ignore_index=True)
    tta = tta.rename(columns={"r2": "tta_r2", "support_size": "support_samples"})

    # ── Join vanilla baseline ───────────────────────────────────────────────────
    merged = tta.merge(vanilla_lookup, on="session_id", how="left")

    n_missing = merged["vanilla_test_r2"].isna().sum()
    if n_missing:
        missing_sids = merged.loc[merged["vanilla_test_r2"].isna(), "session_id"].unique()
        print(f"  WARNING: {n_missing} rows missing vanilla R² for sessions: {missing_sids}")

    merged["improvement_vs_vanilla"] = merged["tta_r2"] - merged["vanilla_test_r2"]

    # ── Select and order columns ────────────────────────────────────────────────
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
