"""
Reusable functions and math-y things go here.
Take no deps on other tbfm modules; this is a leaf.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torcheval.metrics.functional import r2_score


def zscore(data, mean, std):
    """
    data: (batch, time, channel)
    mean: (channel,)
    std: (channel,)
    """
    demeaned = data - mean
    zscored = demeaned / std
    return zscored


def zscore_inv(data, mean, std):
    """
    data: (batch, time, channel)
    mean: (channel,)
    std: (channel,)
    """
    unscaled = data * std
    un_zscored = unscaled + mean
    return un_zscored


def percentile_affine(x, p_low=0.1, p_high=0.9, floor=1e-3):
    # x: [B, T, C] few-shot stim samples
    flat = x.flatten(end_dim=1)
    q = torch.tensor([p_low, 0.5, p_high], device=x.device, dtype=x.dtype)
    qs = torch.quantile(flat, q, dim=0)  # [3, C]
    ql, qm, qh = qs[0], qs[1], qs[2]
    s = (qh - ql).clamp_min(floor * (qh - ql).median().clamp_min(1e-6))
    a = 2.0 / s
    b = -(qh + ql) / 2.0  # Average high and low; that's our new center
    # returns scale a and bias b such that x' = a*(x + b)
    return a, b


class SessionDispatcher:
    # Must be overridden for specific uses; e.g. ["encode", "decode"]
    DISPATCH_METHODS = None

    def __init__(self, instances: dict):
        """
        instances: {session_id: instance}
        """
        self.instances = instances
        self._closed_kwargs = {}  # {method_name: {session_id: {k: value}}}

    def close_kwarg(self, name: str, k, values: dict):
        """
        values: {session_id: v}
        """
        meth_vals = self._closed_kwargs.setdefault(name, {})
        for session_id, v in values.items():
            sid_vals = meth_vals.setdefault(session_id, {})
            sid_vals[k] = v

        # Now reset the closure if it's there
        try:
            delattr(self, name)
        except AttributeError:
            pass

    def get_closed_kwargs(self, name):
        return self._closed_kwargs.get(name, {})

    def __call__(self, x, *args, **kwargs):
        outputs = {}
        ck = self.get_closed_kwargs("__call__")
        for session_id, _x in x.items():
            output = self.instances[session_id](
                _x, *args, **ck.get(session_id, {}), **kwargs
            )
            outputs[session_id] = output
        return outputs

    def attempt_static_dispatch(self, name: str):
        raise AttributeError(f"Cannot find dispatch path for {name}")

    def __getattr__(self, name: str):
        """
        Dispatch calls to individual instances.
        """
        if self.DISPATCH_METHODS is None or name not in self.DISPATCH_METHODS:
            # Here we don't have a per-session argument
            ck = self.get_closed_kwargs(name)

            def dispatcher(*args, **kwargs):
                outputs = {}
                for session_id, instance in self.instances.items():
                    output = getattr(instance, name)(
                        *args, **ck.get(session_id, {}), **kwargs
                    )
                    outputs[session_id] = output
                return outputs

        else:
            # Here we do have a per-session argument
            ck = self.get_closed_kwargs(name)

            def dispatcher(x, *args, **kwargs):
                outputs = {}
                for session_id, _x in x.items():
                    output = getattr(self.instances[session_id], name)(
                        _x, *args, **ck.get(session_id, {}), **kwargs
                    )
                    outputs[session_id] = output
                return outputs

        # Cache the generated dispatcher so future lookups are fast
        setattr(self, name, dispatcher)
        return dispatcher


def flatten(xs):
    for x in xs:
        # Strings and bytes are iterable but usually should be treated as atomic
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


class OptimCollection:
    """
    Collection of named optimizer groups with associated schedulers.

    Usage:
        groups = {
            "bw": {"optimizers": [optim_bw], "schedulers": [sched_bw]},
            "bg": {"optimizers": [optim_bg], "schedulers": [sched_bg]},
            "meta": {"optimizers": [optim_meta]},
            "ae": {"optimizers": [optim_ae1, optim_ae2]},
        }
        optim_collection = OptimCollection(groups)

        # Update all groups
        optim_collection.step()

        # Update only specific groups
        optim_collection.step(only=["bw", "meta"])

        # Update all except specific groups
        optim_collection.step(skip=["bg"])
    """

    def __init__(self, groups: dict):
        """
        Args:
            groups: dict {group_name: {"optimizers": [...], "schedulers": [...]}}
                   "schedulers" is optional and defaults to empty list
        """
        self.groups = {}

        for name, group_data in groups.items():
            optims = group_data.get("optimizers", [])
            scheds = group_data.get("schedulers", [])

            # Ensure lists, filter None
            optims = [o for o in optims if o is not None]
            scheds = [s for s in scheds if s is not None]

            self.groups[name] = {
                "optimizers": optims,
                "schedulers": scheds,
            }

    def zero_grad(self, only=None, skip=None, **kwargs):
        """
        Zero gradients for specified optimizer groups.

        Args:
            only: list of group names (if None, use all groups)
            skip: list of group names to skip (ignored if only is set)
            **kwargs: passed to optimizer.zero_grad()
        """
        for name in self._resolve_groups(only, skip):
            for optim in self.groups[name]["optimizers"]:
                optim.zero_grad(**kwargs)

    def step(self, only=None, skip=None, **kwargs):
        """
        Step optimizers and their schedulers for specified groups.

        Args:
            only: list of group names (if None, use all groups)
            skip: list of group names to skip (ignored if only is set)
            **kwargs: passed to optimizer.step()
        """
        for name in self._resolve_groups(only, skip):
            # Step optimizers
            for optim in self.groups[name]["optimizers"]:
                optim.step(**kwargs)

            # Step schedulers
            for sched in self.groups[name]["schedulers"]:
                sched.step()

    def clip_grad(self, value=1.0, only=None, skip=None):
        """
        Clip gradients for specified optimizer groups.

        Args:
            value: max gradient norm
            only: list of group names (if None, use all groups)
            skip: list of group names to skip (ignored if only is set)
        """
        for name in self._resolve_groups(only, skip):
            for optim in self.groups[name]["optimizers"]:
                for param_group in optim.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            torch.nn.utils.clip_grad_norm_(p, value)

    def _resolve_groups(self, only=None, skip=None):
        """Determine which groups to operate on."""
        if only is not None:
            return only

        all_groups = list(self.groups.keys())
        if skip is not None:
            return [g for g in all_groups if g not in skip]

        return all_groups


def async_copy_sessions_to_device(sessions, device):
    """
    Enqueue H2D copies for many sessions in parallel, one CUDA stream per session.
    Assumes CPU tensors are already pinned (page-locked). Uses non_blocking=True.

    Args:
        sessions: {sid: (x1, x2, ...)}
        device:   target device (e.g., "cuda:0")

    Returns:
        gpu_dict:  {sid: (x1, x2, ...)}  (copies may still be in-flight)
        streams:   {sid: torch.cuda.Stream()} used for the copies
    """

    gpu_dict = {}
    streams = {}
    for sid, d in sessions.items():
        s = torch.cuda.Stream()
        streams[sid] = s

        # If some are already on GPU, we just pass them through
        def _to_dev(t: torch.Tensor):
            if t.device.type == "cuda":
                return t
            # Important: non_blocking=True requires pinned source memory to truly async
            return t.to(device, non_blocking=True)

        with torch.cuda.stream(s):
            d_gpu = tuple(_to_dev(dd.contiguous()) for dd in d)
            gpu_dict[sid] = d_gpu

    return gpu_dict, streams


def wait_streams(streams) -> None:
    """Join all per-session streams with the current (default) stream."""
    cur = torch.cuda.current_stream()
    for s in streams.values():
        cur.wait_stream(s)


def move_batch(batch, device=None):
    if device != "cpu":
        batch, streams = async_copy_sessions_to_device(batch, device=device)
        wait_streams(streams)
    return batch


def iter_loader(iterator, loader, device=None):
    """
    Advance the iterator and return the data.
    """
    try:
        data = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        data = next(iterator)
    return iterator, move_batch(data, device=device)


def rotate_session_from_batch(data, session_id, device=None):
    d = []
    for batch in data:
        session_batch = batch[session_id]
        if not d:
            for _ in range(len(session_batch)):
                d.append([])
        for idx in range(len(session_batch)):
            d[idx].append(session_batch[idx])
    output = tuple(torch.cat(dd, dim=0).to(device) for dd in d)
    return output


def log_grad_norms(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_norm = param.data.norm().item()
            ratio = grad_norm / (weight_norm + 1e-8)  # avoid div by zero
            print(
                f"{name:30s} grad_norm={grad_norm:.4e}, "
                f"weight_norm={weight_norm:.4e}, "
                f"ratio={ratio:.4e}"
            )


def create_outlier_mask(
    data,
    iqr_multiplier=7.0,
    center="median",
    max_outliers_per_trial=5,
    return_counts=False,
):
    """
    Create a boolean mask identifying trials (batch elements) with too many outliers.

    This function detects outlier timepoints by measuring how far each sample deviates
    from the center (median or mean) in terms of IQR. Then it rejects entire trials
    that have more than max_outliers_per_trial outlier timepoints.

    Args:
        data: Tensor of shape (batch, time, channels) or (batch, channels)
            The data to check for outliers (should be normalized)
        iqr_multiplier: float, default=7.0
            Number of IQRs from center beyond which timepoints are outliers
            Typical values: 3.0 (aggressive), 7.0 (moderate), 15.0 (conservative)
        center: str, one of ["median", "mean"], default="median"
            How to compute the center for deviation measurement
        max_outliers_per_trial: int, default=5
            Maximum number of outlier timepoints allowed in a trial
            Trials with more outliers than this are rejected entirely
            Suggested: 5-10 for trials with 50-100 timepoints
        return_counts: bool, default=False
            If True, also return the outlier count per trial

    Returns:
        mask: Boolean tensor of shape (batch,)
            True = keep trial (< max_outliers), False = reject trial (>= max_outliers)
        counts: (optional) Integer tensor of shape (batch,)
            Number of outlier timepoints detected in each trial
            Only returned if return_counts=True

    Example:
        >>> data = torch.randn(100, 164, 50)  # (batch=100 trials, time=164, channels=50)
        >>> mask = create_outlier_mask(data, iqr_multiplier=7.0, max_outliers_per_trial=5)
        >>> clean_data = data[mask]  # Keep only clean trials
        >>> print(f"Kept {mask.sum()}/{len(mask)} trials")
        >>>
        >>> # Or get counts to see how many outliers each trial has
        >>> mask, counts = create_outlier_mask(data, return_counts=True)
        >>> print(f"Max outliers in any trial: {counts.max()}")
    """
    # Handle both 2D and 3D inputs
    if data.ndim == 3:
        batch_size, time_size, n_channels = data.shape
        flat_data = data.flatten(end_dim=1)  # (batch*time, channels)
    elif data.ndim == 2:
        batch_size = data.shape[0]
        time_size = 1
        n_channels = data.shape[1]
        flat_data = data
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {data.shape}")

    # Compute center (median or mean) and IQR across all samples
    if center == "median":
        q = torch.tensor([0.25, 0.5, 0.75], device=data.device, dtype=data.dtype)
        quantiles = torch.quantile(flat_data, q, dim=0)  # [3, channels]
        q25, q50, q75 = quantiles[0], quantiles[1], quantiles[2]
        center_val = q50
        iqr = q75 - q25
    elif center == "mean":
        center_val = flat_data.mean(dim=0)  # [channels]
        q = torch.tensor([0.25, 0.75], device=data.device, dtype=data.dtype)
        quantiles = torch.quantile(flat_data, q, dim=0)  # [2, channels]
        iqr = quantiles[1] - quantiles[0]
    else:
        raise ValueError(f"center must be 'median' or 'mean', got {center}")

    # Compute deviation from center for each timepoint
    deviation = torch.abs(flat_data - center_val)  # (batch*time, channels)

    # Threshold: iqr_multiplier * IQR
    # Add small epsilon to avoid division by zero for constant channels
    threshold = iqr_multiplier * iqr.clamp_min(1e-6)

    # A timepoint is an outlier if ANY channel exceeds the threshold
    is_outlier_timepoint = (deviation > threshold).any(dim=-1)  # (batch*time,)

    # Reshape to (batch, time) and count outliers per trial
    if data.ndim == 3:
        is_outlier_timepoint = is_outlier_timepoint.view(batch_size, time_size)
        outlier_counts = is_outlier_timepoint.sum(dim=1)  # (batch,)
    else:
        # For 2D data, each "trial" is just one sample
        outlier_counts = is_outlier_timepoint.long()

    # Accept trials with <= max_outliers_per_trial outliers
    mask = outlier_counts <= max_outliers_per_trial

    if return_counts:
        return mask, outlier_counts
    else:
        return mask


def apply_outlier_mask_to_batch(batch_dict, masks_dict):
    """
    Apply outlier masks to a batch dictionary, filtering out outlier trials.

    Args:
        batch_dict: dict {session_id: (runway, covariates, targets)}
            Each element is a tuple of tensors with shape (batch, time, channels)
        masks_dict: dict {session_id: mask_tensor}
            Each mask is boolean tensor of shape (batch,)
            True = keep trial, False = reject trial

    Returns:
        filtered_batch: dict {session_id: (runway, covariates, targets)}
            Same structure but with outlier trials removed
        counts: dict {session_id: (kept, total)}
            Number of trials kept vs total for each session
    """
    filtered_batch = {}
    counts = {}

    for session_id, data in batch_dict.items():
        if session_id not in masks_dict:
            # No mask for this session, keep all data
            filtered_batch[session_id] = data
            batch_size = data[0].shape[0]
            counts[session_id] = (batch_size, batch_size)
            continue

        mask = masks_dict[session_id]  # Shape: (batch,)

        # Apply trial-level mask to each component of the tuple
        filtered_data = []
        for component in data:
            # component has shape (batch, time, channels) or (batch, channels)
            # mask has shape (batch,)
            filtered_component = component[mask]
            filtered_data.append(filtered_component)

        filtered_batch[session_id] = tuple(filtered_data)
        total = mask.shape[0]  # Number of trials
        kept = mask.sum().item()  # Number of trials kept
        counts[session_id] = (kept, total)

    return filtered_batch, counts


def filter_batch_outliers(batch_dict, model_norms, cfg):
    """
    Complete outlier filtering pipeline: normalize, detect, filter, and track stats.

    This function encapsulates the entire outlier filtering workflow:
    1. Normalizes batch targets using model normalizers
    2. Detects outliers using IQR criterion from config
    3. Filters out outlier trials
    4. Tracks statistics about filtering

    Args:
        batch_dict: dict {session_id: (runway, covariates, targets)}
            Raw batch data with unnormalized targets
        model_norms: normalizers module with .instances[session_id]
            Normalizers for each session
        cfg: configuration object
            Must have training.use_outlier_filtering, outlier_iqr_multiplier,
            outlier_center, and max_outliers_per_trial

    Returns:
        filtered_batch: dict {session_id: (runway, covariates, targets)}
            Batch with outlier trials removed (targets still unnormalized)
        stats: dict with keys:
            'total_kept': int, total trials kept across all sessions
            'total_samples': int, total trials before filtering
            'per_session': dict {session_id: (kept, total)}

    Example:
        >>> filtered_batch, stats = filter_batch_outliers(batch, model.norms, cfg)
        >>> print(f"Kept {stats['total_kept']}/{stats['total_samples']} trials")
    """
    use_outlier_filtering = cfg.training.use_outlier_filtering

    if not use_outlier_filtering:
        # Return batch unchanged with full counts
        stats = {
            "total_kept": sum(d[0].shape[0] for d in batch_dict.values()),
            "total_samples": sum(d[0].shape[0] for d in batch_dict.values()),
            "per_session": {
                sid: (d[0].shape[0], d[0].shape[0]) for sid, d in batch_dict.items()
            },
        }
        return batch_dict, stats

    # Get outlier filtering parameters from config
    iqr_multiplier = cfg.training.outlier_iqr_multiplier
    center = cfg.training.outlier_center
    max_outliers_per_trial = cfg.training.max_outliers_per_trial

    # Create outlier masks based on normalized targets
    outlier_masks = {}
    for sid, d in batch_dict.items():
        y = d[2]  # targets
        # Normalize for outlier detection
        y_norm = model_norms.instances[sid](y)
        mask = create_outlier_mask(
            y_norm,
            iqr_multiplier=iqr_multiplier,
            center=center,
            max_outliers_per_trial=max_outliers_per_trial,
        )
        outlier_masks[sid] = mask

    # Filter the batch
    filtered_batch, per_session_counts = apply_outlier_mask_to_batch(
        batch_dict, outlier_masks
    )

    # Aggregate statistics
    total_kept = sum(c[0] for c in per_session_counts.values())
    total_samples = sum(c[1] for c in per_session_counts.values())

    stats = {
        "total_kept": total_kept,
        "total_samples": total_samples,
        "per_session": per_session_counts,
    }

    return filtered_batch, stats


def normalize_batch_targets(batch_dict, model_norms):
    """
    Normalize the targets (third element) of each session in a batch dict.

    Args:
        batch_dict: dict {session_id: (runway, covariates, targets)}
            Batch with unnormalized targets
        model_norms: normalizers module with .instances[session_id]
            Normalizers for each session

    Returns:
        normalized_batch: dict {session_id: (runway, covariates, targets_normalized)}
            Batch with normalized targets, other elements unchanged

    Example:
        >>> norm_batch = normalize_batch_targets(batch, model.norms)
    """
    normalized_batch = {}
    for session_id, d in batch_dict.items():
        y_norm = model_norms.instances[session_id](d[2])
        normalized_batch[session_id] = (d[0], d[1], y_norm)
    return normalized_batch


def evaluate_test_batches(
    model,
    data_test,
    embeddings_rest,
    embeddings_stim,
    model_norms,
    cfg,
    device,
    track_per_session_r2=False,
):
    """
    Evaluate model on test data with optional outlier filtering.

    This function runs a complete test evaluation loop:
    1. Iterate through test batches
    2. Apply outlier filtering if enabled
    3. Normalize targets
    4. Run model forward pass
    5. Compute loss and R2 metrics
    6. Aggregate results across batches

    Args:
        model: The model to evaluate (should be in eval mode)
        data_test: DataLoader or iterable of test batches
        embeddings_rest: dict {session_id: embedding} for rest embeddings
        embeddings_stim: dict {session_id: embedding} for stim embeddings
        model_norms: normalizers module with .instances[session_id]
        cfg: configuration object for outlier filtering settings
        device: torch device
        track_per_session_r2: bool, whether to track R2 per session (default False)

    Returns:
        results: dict with keys:
            'loss': float, mean test loss
            'r2': float, mean test R2
            'y_hat': dict {session_id: predictions}, from last batch
            'y_test': dict {session_id: (runway, covariates, targets_norm)}, from last batch
            'per_session_r2': dict {session_id: r2}, only if track_per_session_r2=True
            'outlier_stats': dict or None, outlier filtering statistics

    Example:
        >>> with torch.no_grad():
        >>>     model.eval()
        >>>     results = evaluate_test_batches(model, data_test, ...)
        >>> print(f"Test R2: {results['r2']:.4f}")
    """
    loss_outer = 0
    r2_outer = 0
    test_batch_count = 0
    per_session_r2 = {} if track_per_session_r2 else None

    # Outlier filtering stats
    use_outlier_filtering = cfg.training.use_outlier_filtering
    outlier_total_kept = 0
    outlier_total_samples = 0

    # Will store results from last batch
    y_hat_test = None
    test_batch = None

    for batch in data_test:
        batch = move_batch(batch, device=device)

        # Initialize per-session R2 tracking
        if track_per_session_r2 and per_session_r2 == {}:
            per_session_r2 = {session_id: 0.0 for session_id in batch.keys()}

        # Apply outlier filtering if enabled
        if use_outlier_filtering:
            batch, filter_stats = filter_batch_outliers(batch, model_norms, cfg)
            outlier_total_kept += filter_stats["total_kept"]
            outlier_total_samples += filter_stats["total_samples"]

        # Normalize targets
        batch = normalize_batch_targets(batch, model_norms)

        # Forward pass
        y_hat_test = model(
            batch,
            embeddings_rest=embeddings_rest,
            embeddings_stim=embeddings_stim,
        )

        # Compute loss and R2
        loss = 0
        r2_test = 0
        for session_id, d in batch.items():
            y_test = d[2]
            loss += nn.MSELoss()(y_hat_test[session_id], y_test)
            _r2_test = r2_score(
                y_hat_test[session_id].permute(0, 2, 1).flatten(end_dim=1),
                y_test.permute(0, 2, 1).flatten(end_dim=1),
            )
            r2_test += _r2_test
            if track_per_session_r2:
                per_session_r2[session_id] += _r2_test.item()

        loss /= len(batch)
        r2_test /= len(batch)

        r2_outer += r2_test.item()
        loss_outer += loss.item()
        test_batch_count += 1

        # Save last batch for return
        test_batch = batch

    # Aggregate results
    if test_batch_count > 0:
        r2_final = r2_outer / test_batch_count
        loss_final = loss_outer / test_batch_count
        if track_per_session_r2:
            per_session_r2 = {
                sid: r2 / test_batch_count for sid, r2 in per_session_r2.items()
            }
    else:
        r2_final = 0.0
        loss_final = 0.0
        per_session_r2 = {}

    # Prepare outlier stats
    outlier_stats = None
    if use_outlier_filtering and outlier_total_samples > 0:
        outlier_stats = {
            "total_kept": outlier_total_kept,
            "total_samples": outlier_total_samples,
            "pct_kept": 100.0 * outlier_total_kept / outlier_total_samples,
        }

    results = {
        "loss": loss_final,
        "r2": r2_final,
        "y_hat": y_hat_test,
        "y_test": test_batch,
        "outlier_stats": outlier_stats,
    }

    if track_per_session_r2:
        results["per_session_r2"] = per_session_r2

    return results
