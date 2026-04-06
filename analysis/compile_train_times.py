#!/usr/bin/env python3
"""Compile TTA, random-fold, and vanilla train times into a single CSV.

Sources:
  - random_folds_20260101_211200/timing_log.txt          (run_kind: random_folds_train)
  - random_folds_20260101_211200/tta_results_*/tta_timing_log.txt  (run_kind: tta_cv)
  - vanilla_training_times.csv                            (run_kind: vanilla_train)

R² for vanilla rows is read from state_dependency_results_vanilla_{train_size}/{session}/summary.txt
when those directories exist alongside this script.
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RANDOM_FOLDS_DIR = os.path.join(REPO_ROOT, "random_folds_20260101_211200")
VANILLA_TIMES_CSV = os.path.join(REPO_ROOT, "vanilla_training_times.csv")


# ---------------------------------------------------------------------------
# Fold / TTA timing (from *timing_log.txt files)
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
        return "random_folds_train"
    if base == "tta_timing_log.txt":
        return "tta_cv"
    return "unknown"


def _parse_tta_meta(path: str) -> tuple[Optional[int], Optional[str]]:
    """Return (tta_train_size, tta_run_dir_timestamp) for tta_results_<N>_<ts> dirs."""
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


def collect_timing_files() -> list[str]:
    """Return all *timing_log.txt files from the random_folds directory."""
    pattern = os.path.join(RANDOM_FOLDS_DIR, "**", "*timing_log.txt")
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
# Vanilla training times (from vanilla_training_times.csv)
# ---------------------------------------------------------------------------

RE_VANILLA_SUMMARY_R2 = re.compile(r"^\s*(?P<key>Train|Test) R²:\s*(?P<val>[0-9.]+)\s*$")


def _read_vanilla_r2(train_size: int, session: str) -> tuple[str, str]:
    """Read Train/Test R² from the corresponding state_dependency_results_vanilla dir."""
    summary_path = os.path.join(
        REPO_ROOT,
        f"state_dependency_results_vanilla_{train_size}",
        session,
        "summary.txt",
    )
    if not os.path.exists(summary_path):
        return "", ""
    train_r2 = test_r2 = ""
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            m = RE_VANILLA_SUMMARY_R2.match(line)
            if m:
                if m.group("key") == "Train":
                    train_r2 = m.group("val")
                else:
                    test_r2 = m.group("val")
    return train_r2, test_r2


def _dt_to_ts(dt_str: str) -> str:
    """Convert '2026-01-26 20:51:23' to '20260126_205123'."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y%m%d_%H%M%S")


def build_vanilla_rows() -> list[dict]:
    if not os.path.exists(VANILLA_TIMES_CSV):
        return []

    rows: list[dict] = []
    rel_path = os.path.relpath(VANILLA_TIMES_CSV, REPO_ROOT)

    with open(VANILLA_TIMES_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            train_size = int(rec["support_samples"])
            session = rec["session"]
            duration_s_float = float(rec["duration_seconds"])
            duration_s = round(duration_s_float)
            start_ts = _dt_to_ts(rec["start_time"])
            end_ts = _dt_to_ts(rec["end_time"])
            train_r2, test_r2 = _read_vanilla_r2(train_size, session)

            rows.append(
                {
                    "run_kind": "vanilla_train",
                    "run_label": start_ts,
                    "train_size": train_size,
                    "fold": "",
                    "session": session,
                    "gpu_id": "",
                    "fold_started_at": start_ts,
                    "fold_completed_at": end_ts,
                    "duration_s": duration_s,
                    "duration_hours": duration_s / 3600.0,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "source_file": rel_path,
                    "file_started_at": "",
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
    # Use fold if present, else sort by session name
    fold_raw = r.get("fold", "")
    fold_num = int(fold_raw) if fold_raw != "" else -1
    session = str(r.get("session", ""))
    return (run_kind, size_num, run_label, fold_num, session)


def main() -> None:
    timing_files = collect_timing_files()
    if not timing_files:
        raise SystemExit(f"No '*timing_log.txt' files found under: {RANDOM_FOLDS_DIR}")

    print(f"Found {len(timing_files)} timing file(s):")
    for f in timing_files:
        print(f"  {os.path.relpath(f, REPO_ROOT)}")

    rows = build_rows(timing_files)

    vanilla_rows = build_vanilla_rows()
    if vanilla_rows:
        print(f"\nFound vanilla_training_times.csv — {len(vanilla_rows)} session rows")
        rows.extend(vanilla_rows)
    else:
        print("\nNo vanilla_training_times.csv found, skipping.")

    rows.sort(key=_sort_key)

    out_csv = os.path.join(REPO_ROOT, "all_train_times.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {os.path.relpath(out_csv, REPO_ROOT)}")

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
