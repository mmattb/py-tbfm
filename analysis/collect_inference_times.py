#!/usr/bin/env python3
"""
Collect per-trial inference latency for all multisession folds and vanilla sessions.

Output: all_inference_times.csv
Columns: model_type, train_size, fold, session, device, mean_ms, p50_ms, p99_ms,
         n_iterations, model_dir

Usage:
    python collect_inference_times.py [--iterations N] [--gpu GPU_ID] [--cpu-only]
                                      [--output PATH] [--skip-existing]

Defaults:
    --iterations 500   (per model×device combination)
    --gpu 0
    --output all_inference_times.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

# Import helpers from the single-model timing script
from time_inference import (
    apply_hyperparams,
    load_cfg,
    load_session_data,
    batch_to_timing_loader,
    load_vanilla_model,
    _VanillaTimingWrapper,
    _SingleSessionLoader,
)
import timings
import tbfm.multisession as multisession

REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")

FOLD_BASE = REPO_ROOT / "random_folds_20260101_211200"
VANILLA_SIZES = [500, 1000, 2500, 5000]

CSV_FIELDNAMES = [
    "model_type", "train_size", "fold", "session", "device",
    "mean_ms", "p50_ms", "p99_ms", "n_iterations", "model_dir",
]


# ---------------------------------------------------------------------------
# CSV writer (progressive — opens in append mode per row)
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
# Timing helper
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
# Multisession fold timing
# ---------------------------------------------------------------------------

def time_fold(
    fold_dir: Path,
    session_id: str,
    cfg,
    device_str: str,
    iterations: int,
    compiled: bool = False,
) -> tuple[float, float, float]:
    params = torch.load(fold_dir / "hyperparameters.torch", map_location="cpu")
    embeddings_stim_all: dict = torch.load(fold_dir / "es.torch", map_location="cpu")
    model_file = fold_dir / "model.torch"

    fold_cfg = apply_hyperparams(cfg, params)
    runway = fold_cfg.data.runway if hasattr(fold_cfg.data, "runway") else 20

    d = load_session_data(session_id, unpack_stiminds=False)
    batch = next(iter(d))
    timing_loader = batch_to_timing_loader(batch, session_id)

    ms_model = multisession.build_from_cfg(
        fold_cfg,
        d,
        base_model_path=str(model_file),
        device=device_str,
        quiet=True,
    )
    ms_model.eval()

    emb_rest = multisession.load_rest_embeddings(
        [session_id], in_dir=DATA_DIR, device=device_str
    )
    emb_stim = {session_id: embeddings_stim_all[session_id].to(device_str)}

    if compiled:
        from tbfm._multisession_module import TBFMMultisessionCompiled
        stiminds_ref = timing_loader.stiminds[0:1, runway, :2].to(device_str)
        compiled_ms = TBFMMultisessionCompiled(
            ms_model, session_id, stiminds_ref,
            emb_rest[session_id], emb_stim[session_id],
        )
        wrapper = timings.TimingModuleMultisessionCompiled(compiled_ms, runway, session_id)
    else:
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
# Vanilla session timing
# ---------------------------------------------------------------------------

def time_vanilla(
    model_path: Path,
    session_id: str,
    device_str: str,
    iterations: int,
) -> tuple[float, float, float]:
    d = load_session_data(session_id, unpack_stiminds=False)
    batch = next(iter(d))
    timing_loader = batch_to_timing_loader(batch, session_id)

    model, arch = load_vanilla_model(model_path, device=device_str)
    wrapper = _VanillaTimingWrapper(model, arch["runway"])

    elapsed_ns = timings.time_inference(wrapper, timing_loader, session_id, iterations=iterations)
    return _summarise(elapsed_ns)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_folds(
    writer: CsvWriter,
    devices: list[str],
    iterations: int,
    cfg,
    include_compiled: bool = False,
) -> None:
    # All folds share the same hyperparameters and train_size=5000, so inference
    # time is architecture-identical across folds. Time fold0 as the representative.
    fold_dir = FOLD_BASE / "fold0"
    print(f"\n{'='*60}")
    print(f"Multisession: timing fold0 (representative — all folds identical architecture)")
    print(f"{'='*60}")

    hisi: list[str] = torch.load(fold_dir / "hisi.torch", map_location="cpu")
    session_id = hisi[0]
    fold_name = fold_dir.name
    train_size = 5000

    variants = [("multisession", False)]
    if include_compiled:
        variants.append(("multisession_compiled", True))

    for model_type, compiled in variants:
        for device_str in devices:
            row_proto = {
                "model_type": model_type,
                "train_size": train_size,
                "fold": fold_name,
                "session": session_id,
                "device": device_str,
                "n_iterations": iterations,
                "model_dir": str(fold_dir),
            }
            if writer.already_done(row_proto):
                print(f"  [skip] {model_type} / {fold_name} / {session_id} / {device_str}")
                continue

            print(f"  Timing {model_type} / {fold_name} / {session_id} / {device_str} … ", end="", flush=True)
            try:
                mean_ms, p50_ms, p99_ms = time_fold(
                    fold_dir, session_id, cfg, device_str, iterations, compiled=compiled
                )
                row_proto.update({"mean_ms": f"{mean_ms:.4f}", "p50_ms": f"{p50_ms:.4f}", "p99_ms": f"{p99_ms:.4f}"})
                writer.write(row_proto)
                print(f"mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
            except Exception as e:
                print(f"ERROR: {e}")


def collect_vanilla(
    writer: CsvWriter,
    devices: list[str],
    iterations: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"Vanilla sessions: {len(VANILLA_SIZES)} train sizes")
    print(f"{'='*60}")

    for train_size in VANILLA_SIZES:
        vanilla_dir = REPO_ROOT / f"state_dependency_results_vanilla_{train_size}"
        if not vanilla_dir.exists():
            print(f"  [missing] {vanilla_dir}")
            continue

        session_dirs = sorted(
            p for p in vanilla_dir.iterdir()
            if p.is_dir() and (p / "vanilla_model.torch").exists()
        )
        print(f"\n  train_size={train_size}: {len(session_dirs)} sessions")

        for session_dir in session_dirs:
            session_id = session_dir.name
            model_path = session_dir / "vanilla_model.torch"

            for device_str in devices:
                row_proto = {
                    "model_type": "vanilla",
                    "train_size": train_size,
                    "fold": "",
                    "session": session_id,
                    "device": device_str,
                    "n_iterations": iterations,
                    "model_dir": str(session_dir),
                }
                if writer.already_done(row_proto):
                    print(f"    [skip] {session_id} / {device_str}")
                    continue

                print(f"    Timing {session_id} / {device_str} … ", end="", flush=True)
                try:
                    mean_ms, p50_ms, p99_ms = time_vanilla(
                        model_path, session_id, device_str, iterations
                    )
                    row_proto.update({"mean_ms": f"{mean_ms:.4f}", "p50_ms": f"{p50_ms:.4f}", "p99_ms": f"{p99_ms:.4f}"})
                    writer.write(row_proto)
                    print(f"mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
                except Exception as e:
                    print(f"ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect inference timing for all multisession folds and vanilla sessions"
    )
    parser.add_argument("--iterations", type=int, default=500,
                        help="Timed iterations per model×device (default: 500)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU timing")
    parser.add_argument("--gpu-only", action="store_true", help="Skip CPU timing")
    parser.add_argument("--output", default="all_inference_times.csv",
                        help="Output CSV path (default: all_inference_times.csv)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip rows already in the output CSV (for resuming)")
    parser.add_argument("--folds-only", action="store_true", help="Only time multisession folds")
    parser.add_argument("--vanilla-only", action="store_true", help="Only time vanilla sessions")
    parser.add_argument("--compile", action="store_true",
                        help="Also time compiled multisession model (prerendered bases)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Determine devices to benchmark
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
    print(f"Iterations : {args.iterations} per model×device")

    output_path = Path(args.output)
    writer = CsvWriter(output_path, skip_existing=args.skip_existing)

    # Load Hydra config once
    cfg = load_cfg()

    if not args.vanilla_only:
        collect_folds(writer, devices, args.iterations, cfg, include_compiled=args.compile)

    if not args.folds_only:
        collect_vanilla(writer, devices, args.iterations)

    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    main()
