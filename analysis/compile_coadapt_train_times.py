#!/usr/bin/env python3
"""Compile coadapt training and TTA times into a single CSV.

Same log format as compile_train_times.py but targets the coadapt folds directory.

Sources:
  - random_folds_coadapt_*/timing_log.txt          (run_kind: coadapt_train)
  - random_folds_coadapt_*/tta_results_*/tta_timing_log.txt  (run_kind: coadapt_tta_cv)

Usage:
    python compile_coadapt_train_times.py [--folds-dir DIR] [--output PATH]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDS_DIR = os.path.join(REPO_ROOT, "random_folds_coadapt_20260225_015603")


# ---------------------------------------------------------------------------
# Fold / TTA timing (identical log format to MAML)
# ---------------------------------------------------------------------------

@dataclass
class FoldTiming:
    fold: int
    fold_started_at: Optional[str] = None
    fold_completed_at: Optional[str] = None
    duration_s: Optional[int] = None
    gpu_id: Optional[int] = None
    train_r2: Optional[str] = None
    test_r2: Optional[str] = None


RE_HEADER_STARTED = re.compile(r"^Started at:\s+(?P<ts>\d{8}_\d{6})\s*$")
RE_FOLD_STARTED = re.compile(
    r"^Fold\s+(?P<fold>\d+):\s+STARTED\s+at\s+(?P<ts>\d{8}_\d{6})(?:\s+on\s+GPU\s+(?P<gpu>\d+))?\s*$"
)
RE_FOLD_COMPLETED = re.compile(
    r"^Fold\s+(?P<fold>\d+):\s+COMPLETED\s+at\s+(?P<ts>\d{8}_\d{6})\s+\(duration:\s+(?P<dur>\d+)s"
    r"(?:,\s+train_r2:\s*(?P<train_r2>[^,]+),\s+test_r2:\s*(?P<test_r2>[^)]+))?\)\s*$"
)
RE_TTA_DIR = re.compile(r"tta_results_(?P<train_size>\d+)_(?P<ts>\d{8}_\d{6})$")


def _run_kind_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base == "timing_log.txt":
        return "coadapt_train"
    if base == "tta_timing_log.txt":
        return "coadapt_tta_cv"
    return "unknown"


def _parse_tta_meta(path: str) -> tuple[Optional[int], Optional[str]]:
    parent = os.path.basename(os.path.dirname(path))
    m = RE_TTA_DIR.match(parent)
    if not m:
        return None, None
    return int(m.group("train_size")), m.group("ts")


def parse_timing_file(path: str) -> tuple[Optional[str], dict[int, FoldTiming]]:
    file_started_at: Optional[str] = None
    folds: dict[int, FoldTiming] = {}

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = RE_HEADER_STARTED.match(line)
            if m:
                file_started_at = m.group("ts")
                continue

            m = RE_FOLD_STARTED.match(line)
            if m:
                fold = int(m.group("fold"))
                ft = folds.get(fold) or FoldTiming(fold=fold)
                ft.fold_started_at = m.group("ts")
                gpu = m.group("gpu")
                ft.gpu_id = int(gpu) if gpu is not None else ft.gpu_id
                folds[fold] = ft
                continue

            m = RE_FOLD_COMPLETED.match(line)
            if m:
                fold = int(m.group("fold"))
                ft = folds.get(fold) or FoldTiming(fold=fold)
                ft.fold_completed_at = m.group("ts")
                ft.duration_s = int(m.group("dur"))
                train_r2 = m.group("train_r2")
                test_r2 = m.group("test_r2")
                ft.train_r2 = train_r2.strip() if train_r2 is not None else ft.train_r2
                ft.test_r2 = test_r2.strip() if test_r2 is not None else ft.test_r2
                folds[fold] = ft
                continue

    return file_started_at, folds


def collect_timing_files(folds_dir: str) -> list[str]:
    pattern = os.path.join(folds_dir, "**", "*timing_log.txt")
    return sorted(glob(pattern, recursive=True))


def build_rows(timing_files: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in timing_files:
        run_kind = _run_kind_from_path(path)
        tta_train_size, tta_dir_started_at = _parse_tta_meta(path)
        file_started_at, fold_map = parse_timing_file(path)

        rel_path = os.path.relpath(path, REPO_ROOT)
        run_label = tta_dir_started_at or file_started_at or ""

        for fold, ft in fold_map.items():
            if ft.duration_s is None and ft.fold_completed_at is None and ft.fold_started_at is None:
                continue
            rows.append(
                {
                    "run_kind": run_kind,
                    "run_label": run_label,
                    "train_size": tta_train_size if tta_train_size is not None else "",
                    "fold": fold,
                    "session": "",
                    "gpu_id": ft.gpu_id if ft.gpu_id is not None else "",
                    "fold_started_at": ft.fold_started_at or "",
                    "fold_completed_at": ft.fold_completed_at or "",
                    "duration_s": ft.duration_s if ft.duration_s is not None else "",
                    "duration_hours": (ft.duration_s / 3600.0) if ft.duration_s is not None else "",
                    "train_r2": ft.train_r2 or "",
                    "test_r2": ft.test_r2 or "",
                    "source_file": rel_path,
                    "file_started_at": file_started_at or "",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "run_kind",
    "run_label",
    "train_size",
    "fold",
    "session",
    "gpu_id",
    "fold_started_at",
    "fold_completed_at",
    "duration_s",
    "duration_hours",
    "train_r2",
    "test_r2",
    "source_file",
    "file_started_at",
]


def _sort_key(r: dict) -> tuple:
    run_kind = str(r.get("run_kind", ""))
    size = r.get("train_size", "")
    try:
        size_num = int(size) if size != "" else -1
    except Exception:
        size_num = -1
    run_label = str(r.get("run_label", ""))
    fold_raw = r.get("fold", "")
    fold_num = int(fold_raw) if fold_raw != "" else -1
    session = str(r.get("session", ""))
    return (run_kind, size_num, run_label, fold_num, session)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile coadapt training and TTA times into a CSV"
    )
    parser.add_argument(
        "--folds-dir", default=DEFAULT_FOLDS_DIR,
        help=f"Path to coadapt random folds directory (default: {DEFAULT_FOLDS_DIR})"
    )
    parser.add_argument(
        "--output", default=os.path.join(REPO_ROOT, "all_coadapt_train_times.csv"),
        help="Output CSV path (default: all_coadapt_train_times.csv)"
    )
    args = parser.parse_args()

    timing_files = collect_timing_files(args.folds_dir)
    if not timing_files:
        raise SystemExit(f"No '*timing_log.txt' files found under: {args.folds_dir}")

    print(f"Found {len(timing_files)} timing file(s):")
    for f in timing_files:
        print(f"  {os.path.relpath(f, REPO_ROOT)}")

    rows = build_rows(timing_files)
    rows.sort(key=_sort_key)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {os.path.relpath(args.output, REPO_ROOT)}")

    summary: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["run_kind"], r["train_size"])
        if r["duration_s"] != "":
            summary[key].append(float(r["duration_s"]))
    print("\nSummary (mean duration per group):")
    for (kind, size), durations in sorted(summary.items()):
        label = f"{kind}" + (f" train_size={size}" if size != "" else "")
        mean_h = sum(durations) / len(durations) / 3600
        total_h = sum(durations) / 3600
        print(f"  {label}: n={len(durations)}, mean={mean_h:.2f}h, total={total_h:.2f}h")


if __name__ == "__main__":
    main()
