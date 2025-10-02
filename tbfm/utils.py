"""
Reusable functions and math-y things go here.
Take no deps on other tbfm modules; this is a leaf.
"""

from collections.abc import Iterable

import torch


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
    def __init__(self, collection, schedulers=()):
        c = flatten(collection)
        self.optims = tuple(c)
        self.schedulers = tuple(flatten(schedulers))

    def dispatch(self, attr, **kwargs):
        for optim in self.optims:
            getattr(optim, attr)(**kwargs)

    def zero_grad(self, **kwargs):
        return self.dispatch("zero_grad", **kwargs)

    def step(self, **kwargs):
        ret = self.dispatch("step", **kwargs)
        for sched in self.schedulers:
            sched.step()
        return ret

    def clip_grad(self, value=1.0):
        for optim in self.optims:
            for param_group in optim.param_groups:
                for p in param_group["params"]:
                    if p.grad is None:
                        continue
                    torch.nn.utils.clip_grad_norm_(p, value)


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
