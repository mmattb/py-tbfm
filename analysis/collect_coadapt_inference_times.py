#!/usr/bin/env python3
"""
Collect per-trial inference latency for all coadapt random-folds folds.

Coadapt models are loaded via multisession.load_model_components (tbfm.torch /
ae.torch / norms.torch in the fold directory) rather than a single model.torch.

Output: all_coadapt_inference_times.csv
Columns: model_type, train_size, fold, session, device, mean_ms, p50_ms, p99_ms,
         n_iterations, model_dir

Usage:
    python collect_coadapt_inference_times.py [--iterations N] [--gpu GPU_ID] [--cpu-only]
                                              [--output PATH] [--skip-existing]
                                              [--folds-dir DIR]

Defaults:
    --iterations 500
    --gpu 0
    --output all_coadapt_inference_times.csv
    --folds-dir random_folds_coadapt_20260225_015603
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

from time_inference import (
    apply_hyperparams,
    load_session_data,
    batch_to_timing_loader,
)
import timings
import tbfm.multisession as multisession

REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")

DEFAULT_FOLD_BASE = REPO_ROOT / "random_folds_coadapt_20260225_015603"
TRAIN_SIZE = 5000  # coadapt folds all use train_size=5000


def load_cfg():
    from copy import deepcopy
    from hydra import initialize_config_dir, compose
    conf_dir = REPO_ROOT / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config")
    return cfg

CSV_FIELDNAMES = [
    "model_type", "train_size", "fold", "session", "device",
    "mean_ms", "p50_ms", "p99_ms", "n_iterations", "model_dir",
]


# ---------------------------------------------------------------------------
# CSV writer (progressive — same as collect_inference_times.py)
# ---------------------------------------------------------------------------

class CsvWriter:
    def __init__(self, path: Path, skip_existing: bool):
        self.path = path
        self.skip_existing = skip_existing
        self.existing: set[tuple] = set()

        if path.exists() and skip_existing:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.existing.add(self._key(row))
            print(f"Loaded {len(self.existing)} existing rows from {path}")

        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

    @staticmethod
    def _key(row: dict) -> tuple:
        return (row["model_type"], str(row["train_size"]), row["fold"],
                row["session"], row["device"])

    def already_done(self, row: dict) -> bool:
        return self.skip_existing and self._key(row) in self.existing

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerow(row)
        self.existing.add(self._key(row))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _summarise(elapsed_ns: list[int]) -> tuple[float, float, float]:
    sv = sorted(elapsed_ns)
    mean_ms = sum(sv) / len(sv) / 1e6
    p50_ms = _percentile(sv, 0.50) / 1e6
    p99_ms = _percentile(sv, 0.99) / 1e6
    return mean_ms, p50_ms, p99_ms


# ---------------------------------------------------------------------------
# Coadapt fold timing
# ---------------------------------------------------------------------------

def time_coadapt_fold(
    fold_dir: Path,
    session_id: str,
    cfg,
    device_str: str,
    iterations: int,
) -> tuple[float, float, float]:
    params = torch.load(fold_dir / "hyperparameters.torch", map_location="cpu")
    embeddings_stim_all: dict = torch.load(fold_dir / "es.torch", map_location="cpu")

    fold_cfg = apply_hyperparams(cfg, params)
    runway = fold_cfg.data.runway if hasattr(fold_cfg.data, "runway") else 20

    d = load_session_data(session_id, unpack_stiminds=False)
    batch = next(iter(d))
    timing_loader = batch_to_timing_loader(batch, session_id)

    # Coadapt: build without a base model path, then load per-session components
    ms_model = multisession.build_from_cfg(
        fold_cfg,
        d,
        base_model_path=None,
        device=device_str,
        quiet=True,
    )
    multisession.load_model_components(fold_dir, ms_model, device=device_str)
    ms_model.eval()

    emb_rest = multisession.load_rest_embeddings(
        [session_id], in_dir=DATA_DIR, device=device_str
    )
    emb_stim = {session_id: embeddings_stim_all[session_id].to(device_str)}

    wrapper = timings.TimingModuleMultisession(
        ms_model,
        runway,
        emb_rest,
        emb_stim,
        stimdim=2,
    )

    elapsed_ns = timings.time_inference(wrapper, timing_loader, session_id, iterations=iterations)
    return _summarise(elapsed_ns)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_folds(
    fold_base: Path,
    writer: CsvWriter,
    devices: list[str],
    iterations: int,
    cfg,
) -> None:
    # All folds have identical architecture; time fold0 as the representative.
    fold_dir = fold_base / "fold0"
    if not fold_dir.exists():
        print(f"  [missing] {fold_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Coadapt: timing fold0 (representative — all folds identical architecture)")
    print(f"{'='*60}")

    hisi: list[str] = torch.load(fold_dir / "hisi.torch", map_location="cpu")
    session_id = hisi[0]
    fold_name = fold_dir.name

    for device_str in devices:
        row_proto = {
            "model_type": "coadapt",
            "train_size": TRAIN_SIZE,
            "fold": fold_name,
            "session": session_id,
            "device": device_str,
            "n_iterations": iterations,
            "model_dir": str(fold_dir),
        }
        if writer.already_done(row_proto):
            print(f"  [skip] {fold_name} / {session_id} / {device_str}")
            continue

        print(f"  Timing {fold_name} / {session_id} / {device_str} … ", end="", flush=True)
        try:
            mean_ms, p50_ms, p99_ms = time_coadapt_fold(
                fold_dir, session_id, cfg, device_str, iterations
            )
            row_proto.update({
                "mean_ms": f"{mean_ms:.4f}",
                "p50_ms": f"{p50_ms:.4f}",
                "p99_ms": f"{p99_ms:.4f}",
            })
            writer.write(row_proto)
            print(f"mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
        except Exception as e:
            print(f"ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect inference timing for all coadapt random-folds folds"
    )
    parser.add_argument("--iterations", type=int, default=500,
                        help="Timed iterations per fold×device (default: 500)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU timing")
    parser.add_argument("--gpu-only", action="store_true", help="Skip CPU timing")
    parser.add_argument("--output", default="all_coadapt_inference_times.csv",
                        help="Output CSV path (default: all_coadapt_inference_times.csv)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip rows already in the output CSV (for resuming)")
    parser.add_argument("--folds-dir", default=str(DEFAULT_FOLD_BASE),
                        help=f"Path to coadapt folds directory (default: {DEFAULT_FOLD_BASE})")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    devices: list[str] = []
    if not args.gpu_only:
        devices.append("cpu")
    if not args.cpu_only:
        gpu_str = f"cuda:{args.gpu}"
        if torch.cuda.is_available():
            devices.append(gpu_str)
        else:
            print("CUDA not available — GPU timing skipped.")

    if not devices:
        print("No devices to benchmark. Check --cpu-only / --gpu-only flags.")
        sys.exit(1)

    print(f"Devices    : {devices}")
    print(f"Iterations : {args.iterations} per fold×device")

    output_path = Path(args.output)
    writer = CsvWriter(output_path, skip_existing=args.skip_existing)

    cfg = load_cfg()

    collect_folds(Path(args.folds_dir), writer, devices, args.iterations, cfg)

    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    main()
