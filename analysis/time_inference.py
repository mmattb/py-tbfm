#!/usr/bin/env python3
"""
Measure per-trial inference latency for multisession and vanilla TBFM models.

Usage:
    python time_inference.py [fold_dir] [--session SESSION_ID] [--gpu GPU_ID]
                             [--iterations N] [--cpu-only]
                             [--vanilla-dir STATE_DEP_VANILLA_DIR]

Defaults to random_folds_20260101_211200/fold0 and the first held-in session.
Add --vanilla-dir to also time the vanilla (single-session) model, e.g.:
    python time_inference.py --vanilla-dir state_dependency_results_vanilla_500
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

import timings
import tbfm.multisession as multisession
from tbfm.tbfm import TBFM as TBFMModel

DATA_DIR = os.getenv("TBFM_DATA_DIR", "/var/data/opto-coproc/")
REPO_ROOT = Path(__file__).parent.resolve()
DEFAULT_FOLD_DIR = REPO_ROOT / "random_folds_20260101_211200" / "fold0"


# ---------------------------------------------------------------------------
# Config helpers (mirrors tta_testing.py clone_cfg_for_eval + param application)
# ---------------------------------------------------------------------------

def load_cfg() -> object:
    conf_dir = REPO_ROOT / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config")
    return cfg


def apply_hyperparams(cfg, params: dict) -> object:
    """Apply saved hyperparameters to a Hydra config (struct-mode safe).

    After resolve=True, Hydra interpolations like ${meta.is_basis_residual} are
    baked into their resolved values, so we must update both the source field
    (e.g. cfg.meta.is_basis_residual) AND the resolved copy in cfg.tbfm.module.
    This mirrors the pattern in tta_testing.py's worker.
    """
    cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(cfg, resolve=True)))
    OmegaConf.set_struct(cfg, False)

    if params.get("latent_dim") is not None:
        cfg.latent_dim = params["latent_dim"]
        cfg.ae.module.latent_dim = params["latent_dim"]
        cfg.tbfm.module.in_dim = params["latent_dim"]
    if params.get("num_bases") is not None:
        cfg.tbfm.module.num_bases = params["num_bases"]
    if params.get("basis_residual_rank") is not None:
        cfg.meta.basis_residual_rank = params["basis_residual_rank"]
        cfg.tbfm.module.basis_residual_rank = params["basis_residual_rank"]
    if params.get("residual_mlp_hidden") is not None:
        cfg.meta.residual_mlp_hidden = params["residual_mlp_hidden"]
        cfg.tbfm.module.residual_mlp_hidden = params["residual_mlp_hidden"]
    if params.get("embed_dim_stim") is not None:
        cfg.tbfm.module.embed_dim_stim = params["embed_dim_stim"]
    if params.get("is_basis_residual") is not None:
        cfg.meta.is_basis_residual = params["is_basis_residual"]
        cfg.tbfm.module.is_basis_residual = params["is_basis_residual"]  # resolved copy
    if params.get("use_two_stage") is not None:
        cfg.ae.use_two_stage = params["use_two_stage"]

    # Eval-time AE config: warm-start as identity, no co-adaptation
    cfg.ae.training.coadapt = False
    cfg.ae.warm_start_is_identity = True

    OmegaConf.set_struct(cfg, True)
    return cfg


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class _SingleSessionLoader:
    """Minimal dataloader shim expected by timings.time_inference.

    time_inference calls next(iter(dataloader)) and expects (x, stiminds).
    """
    def __init__(self, x: torch.Tensor, stiminds: torch.Tensor):
        self.x = x
        self.stiminds = stiminds
        self.session_ids = None  # not needed by time_inference

    def __iter__(self):
        yield (self.x, self.stiminds)


def load_session_data(session_id: str, batch_size: int = 7400, unpack_stiminds: bool = False):
    """Load data for a single session.

    unpack_stiminds=False (default): packed 3-D (batch, time, 3) — for multisession wrapper.
    unpack_stiminds=True: unpacked 2-D (batch, 2) without clock vec — for vanilla wrapper.
    """
    d, _ = multisession.load_stim_batched(
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        unpack_stiminds=unpack_stiminds,
        held_in_session_ids=[session_id],
        batch_size=batch_size,
        num_held_out_sessions=0,
    )
    return d


def batch_to_timing_loader(batch: dict, session_id: str) -> _SingleSessionLoader:
    """Convert a multisession batch dict to the simple (x, stiminds) format."""
    runway_t, stiminds, y = batch[session_id]
    full_x = torch.cat([runway_t, y], dim=1)   # (batch, runway+trial_len, channels)
    return _SingleSessionLoader(full_x, stiminds)


# ---------------------------------------------------------------------------
# Vanilla model helpers
# ---------------------------------------------------------------------------

class _VanillaTimingWrapper(timings.TimingModule):
    """Wrap a vanilla (single-session) TBFM for use with timings.time_inference.

    Data arrives as packed 3-D stiminds (batch, time, 3).  Vanilla training used
    unpack_stiminds=True, which produces (batch, 2) by taking the pulse-position
    dims (columns 1:) at a single timepoint.  We replicate that here by slicing
    stiminds[:, runway, 1:] to drop the clock vector and the time axis.
    """

    def forward(self, session_id, x, stiminds):
        runway_t = x[:, : self.runway, :]
        # stiminds: (batch, time, 3) — slice at runway, drop clock col 0
        stiminds_2d = stiminds[:, self.runway, 1:]   # (batch, 2)
        return self.model(runway_t, stiminds_2d)


def _infer_vanilla_arch(state_dict: dict) -> dict:
    """Derive vanilla TBFM architecture from saved state dict shapes."""
    in_dim = state_dict["normalizer.mean"].shape[1]
    bw = state_dict["basis_weighting.weight"]   # [in_dim*num_bases, in_dim*runway]
    il = state_dict["bases.in_layer.weight"]    # [latent_dim, stimdim-1]
    ol = state_dict["bases.out_layer.weight"]   # [trial_len*num_bases, latent_dim]
    num_hiddens = sum(1 for k in state_dict if k.startswith("bases.hiddens.") and k.endswith(".weight"))
    num_bases = bw.shape[0] // in_dim
    runway = bw.shape[1] // in_dim
    latent_dim = il.shape[0]
    trial_len = ol.shape[0] // num_bases
    stimdim = il.shape[1] + 1  # Bases subtracts 1 for clock vec in __init__
    return dict(
        in_dim=in_dim,
        num_bases=num_bases,
        runway=runway,
        latent_dim=latent_dim,
        trial_len=trial_len,
        stimdim=stimdim,
        basis_depth=num_hiddens,
    )


def load_vanilla_model(model_path: Path, device: str = "cpu") -> TBFMModel:
    """Instantiate a vanilla TBFM and load its saved weights."""
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    arch = _infer_vanilla_arch(state_dict)

    # batchy is only used to init the normalizer; state_dict will overwrite those params.
    dummy_batchy = torch.zeros(2, arch["trial_len"], arch["in_dim"])

    model = TBFMModel(
        in_dim=arch["in_dim"],
        stimdim=arch["stimdim"],
        runway=arch["runway"],
        num_bases=arch["num_bases"],
        trial_len=arch["trial_len"],
        batchy=dummy_batchy,
        latent_dim=arch["latent_dim"],
        basis_depth=arch["basis_depth"],
        zscore=True,
        use_meta_learning=False,
        device=device,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, arch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_timing(
    fold_dir: Path,
    session_id: str | None,
    gpu_id: int | None,
    iterations: int,
    cpu_only: bool,
    compile: bool = False,
) -> None:
    fold_dir = Path(fold_dir)

    # --- Load fold metadata ---
    params = torch.load(fold_dir / "hyperparameters.torch", map_location="cpu")
    hisi: list[str] = torch.load(fold_dir / "hisi.torch", map_location="cpu")
    embeddings_stim_all: dict = torch.load(fold_dir / "es.torch", map_location="cpu")
    model_file = fold_dir / "model.torch"

    if session_id is None:
        session_id = hisi[0]

    if session_id not in hisi:
        raise ValueError(f"Session '{session_id}' not in fold's held-in sessions.\n"
                         f"Available: {hisi}")

    print(f"Fold dir   : {fold_dir}")
    print(f"Session    : {session_id}")
    print(f"Iterations : {iterations}")
    print(f"Hyperparams: {params}")
    print()

    # --- Hydra config ---
    cfg = apply_hyperparams(load_cfg(), params)
    runway = cfg.data.runway if hasattr(cfg.data, "runway") else 20

    # --- Data ---
    print(f"Loading data for {session_id}…")
    d = load_session_data(session_id, unpack_stiminds=False)
    batch = next(iter(d))
    timing_loader = batch_to_timing_loader(batch, session_id)
    print(f"  full_x shape   : {timing_loader.x.shape}")
    print(f"  stiminds shape : {timing_loader.stiminds.shape}")
    print()

    results: dict[str, float] = {}

    def _time_on(device_str: str, label: str, compiled: bool = False) -> float:
        print(f"Building model on {device_str}…")
        ms_model = multisession.build_from_cfg(
            cfg,
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
            print("  Prerendering bases (compiled inference)…")
            from tbfm._multisession_module import TBFMMultisessionCompiled
            # Use the first trial's stim descriptor (at the runway timepoint) as reference
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
                stimdim=2,  # packed data has dim 3; training used unpack_stiminds=True (dim 2)
            )

        compile_tag = "(compiled)" if compiled else ""
        print(f"Timing inference {compile_tag} on {device_str} ({iterations} iterations)…")
        elapsed_ns = timings.time_inference(wrapper, timing_loader, session_id, iterations=iterations)
        mean_ms = sum(elapsed_ns) / len(elapsed_ns) / 1e6
        p50_ms = sorted(elapsed_ns)[len(elapsed_ns) // 2] / 1e6
        p99_ms = sorted(elapsed_ns)[int(len(elapsed_ns) * 0.99)] / 1e6
        print(f"  {label}: mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
        results[label] = mean_ms
        return mean_ms

    _time_on("cpu", "Multisession CPU")
    if compile:
        _time_on("cpu", "Multisession CPU (compiled)", compiled=True)

    if not cpu_only:
        device_str = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"
        if torch.cuda.is_available():
            _time_on(device_str, f"Multisession GPU ({device_str})")
            if compile:
                _time_on(device_str, f"Multisession GPU ({device_str}) (compiled)", compiled=True)
        else:
            print("CUDA not available, skipping GPU timing.")

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    for label, ms in results.items():
        print(f"  {label:35s}: {ms:.3f} ms/trial")


def run_vanilla_timing(
    vanilla_dir: Path,
    session_id: str | None,
    gpu_id: int | None,
    iterations: int,
    cpu_only: bool,
    results: dict,
) -> None:
    vanilla_dir = Path(vanilla_dir)

    # Pick session: use provided, or first session subdir that has a model file
    if session_id is not None:
        model_path = vanilla_dir / session_id / "vanilla_model.torch"
        if not model_path.exists():
            raise FileNotFoundError(f"No vanilla model for session '{session_id}' in {vanilla_dir}")
    else:
        candidates = sorted(
            p for p in vanilla_dir.iterdir()
            if p.is_dir() and (p / "vanilla_model.torch").exists()
        )
        if not candidates:
            raise FileNotFoundError(f"No session dirs with vanilla_model.torch in {vanilla_dir}")
        session_id = candidates[0].name
        model_path = candidates[0] / "vanilla_model.torch"

    print(f"Vanilla dir : {vanilla_dir}")
    print(f"Session     : {session_id}")
    print()

    # Load data in packed 3-D format; the wrapper extracts 2-D stiminds at forward time.
    print(f"Loading data for {session_id}…")
    d = load_session_data(session_id, unpack_stiminds=False)
    batch = next(iter(d))
    timing_loader = batch_to_timing_loader(batch, session_id)
    print(f"  full_x shape   : {timing_loader.x.shape}")
    print(f"  stiminds shape : {timing_loader.stiminds.shape}")
    print()

    def _time_vanilla_on(device_str: str, label: str) -> float:
        print(f"Loading vanilla model on {device_str}…")
        model, arch = load_vanilla_model(model_path, device=device_str)
        print(f"  arch: {arch}")

        wrapper = _VanillaTimingWrapper(model, arch["runway"])

        print(f"Timing vanilla inference on {device_str} ({iterations} iterations)…")
        elapsed_ns = timings.time_inference(wrapper, timing_loader, session_id, iterations=iterations)
        mean_ms = sum(elapsed_ns) / len(elapsed_ns) / 1e6
        p50_ms = sorted(elapsed_ns)[len(elapsed_ns) // 2] / 1e6
        p99_ms = sorted(elapsed_ns)[int(len(elapsed_ns) * 0.99)] / 1e6
        print(f"  {label}: mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
        results[label] = mean_ms
        return mean_ms

    _time_vanilla_on("cpu", "Vanilla CPU")

    if not cpu_only:
        device_str = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"
        if torch.cuda.is_available():
            _time_vanilla_on(device_str, f"Vanilla GPU ({device_str})")
        else:
            print("CUDA not available, skipping GPU timing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time multisession and vanilla TBFM inference latency")
    parser.add_argument(
        "fold_dir",
        nargs="?",
        default=str(DEFAULT_FOLD_DIR),
        help=f"Path to fold directory (default: {DEFAULT_FOLD_DIR})",
    )
    parser.add_argument("--session", default=None, help="Session ID to use (default: first available)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: 0)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of timed iterations (default: 1000)")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU timing")
    parser.add_argument("--compile", action="store_true", help="Also time torch.compile'd model")
    parser.add_argument(
        "--vanilla-dir",
        default=None,
        help="Path to a state_dependency_results_vanilla_* dir to also time the vanilla model",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    all_results: dict[str, float] = {}

    run_timing(
        fold_dir=args.fold_dir,
        session_id=args.session,
        gpu_id=args.gpu,
        iterations=args.iterations,
        cpu_only=args.cpu_only,
        compile=args.compile,
    )

    if args.vanilla_dir:
        print()
        run_vanilla_timing(
            vanilla_dir=args.vanilla_dir,
            session_id=args.session,
            gpu_id=args.gpu,
            iterations=args.iterations,
            cpu_only=args.cpu_only,
            results=all_results,
        )
        print()
        print("=" * 50)
        print("Vanilla Summary")
        print("=" * 50)
        for label, ms in all_results.items():
            print(f"  {label:35s}: {ms:.3f} ms/trial")


if __name__ == "__main__":
    main()
