import time

import torch
from torch.utils.data import DataLoader


class TimingModule:
    def __init__(self, model, runway):
        self.model = model
        self.runway = runway

    @property
    def device(self):
        return self.model.device

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, session_id, x, stiminds):
        raise NotImplementedError()


class TimingModuleTFMUncompiled(TimingModule):
    def forward(self, session_id, x, stiminds):
        runways = x[:, : self.runway, :]
        stiminds = stiminds[:, self.runway :, :]
        yhat = self.model(runways, stiminds)
        return yhat


class TimingModuleTFMCompiled(TimingModule):
    def forward(self, session_id, x, stiminds):
        runways = x[:, : self.runway, :]
        stiminds = stiminds[:, self.runway :, :]
        yhat = self.model(runways)
        return yhat


class TimingModuleTFMCompiled2(TimingModule):
    def forward(self, session_id, x, stiminds):
        runways = x[:, : self.runway, :]
        stiminds = stiminds[:, self.runway :, :]
        yhat = self.model.forward2(runways)
        return yhat


class TimingModuleLSTM(TimingModule):
    def forward(self, session_id, x, stiminds):
        timelen = x.shape[1]
        pred_steps = timelen - self.runway - 1
        y = x[:, 1:, :]
        x = x[:, :-1, :]
        batch = ((session_id,), (x,), (stiminds,), (y,))

        yhats = self.model.forward_inference(batch, pred_steps=pred_steps)
        yhat = yhats[0]
        return yhat


class TimingModuleMultisession(TimingModule):
    def __init__(self, model, runway, embeddings_rest, embeddings_stim=None, stimdim=None):
        super().__init__(model, runway)
        self.embeddings_rest = embeddings_rest
        self.embeddings_stim = embeddings_stim
        self.stimdim = stimdim  # If set, truncate stiminds to this dimension

    def forward(self, session_id, x, stiminds):
        # Prepare data in the format expected by multisession model
        runways = x[:, : self.runway, :]
        
        # Handle both packed (batch, time, stimdim) and unpacked (batch, stimdim) formats
        if stiminds.ndim == 3:
            # Packed format: take the first timestep after runway
            stiminds_unpacked = stiminds[:, self.runway, :]
        elif stiminds.ndim == 2:
            # Already unpacked
            stiminds_unpacked = stiminds
        else:
            raise ValueError(f"Unexpected stiminds shape: {stiminds.shape}")
        
        # Truncate stiminds if needed (for models trained with different stimdim)
        if self.stimdim is not None:
            stiminds_unpacked = stiminds_unpacked[:, :self.stimdim]
        
        y = x[:, self.runway :, :]
        
        data = {session_id: (runways, stiminds_unpacked, y)}
        
        # Get embeddings for this session
        embeddings_rest = {session_id: self.embeddings_rest[session_id]}
        embeddings_stim = None
        if self.embeddings_stim is not None:
            embeddings_stim = {session_id: self.embeddings_stim[session_id]}
        
        yhat = self.model(data, embeddings_rest=embeddings_rest, embeddings_stim=embeddings_stim)
        return yhat[session_id]


class TimingModuleMultisessionCompiled(TimingModule):
    """Timing wrapper for TBFMMultisessionCompiled.

    Passes only the runway to the compiled model; stiminds are ignored because
    the bases matrix was prerendered at compile time.
    """

    def __init__(self, compiled_model, runway, session_id):
        super().__init__(compiled_model, runway)
        self.session_id = session_id

    def forward(self, session_id, x, stiminds):
        runway = x[:, : self.runway, :]
        y = x[:, self.runway :, :]
        data = {session_id: (runway, None, y)}
        yhat = self.model(data)
        return yhat[session_id]


class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter_ns()
        self.elapsed_time = self.end_time - self.start_time


def _time_inference_on_cpu(model, session_id, x, stiminds):
    timer = Timer()
    with timer:
        _ = model(session_id, x, stiminds)

    return timer.elapsed_time


def _time_inference_on_device(model, session_id, x, stiminds, device):
    timer = Timer()
    with timer:
        x = x.to(device)
        yhat = model(session_id, x, stiminds)
        yhat.to(device)

    return timer.elapsed_time


def time_inference(model, dataloader, session_id, iterations=10000):
    batch = next(iter(dataloader))
    x = batch[0]
    dset_size = x.shape[0]
    stiminds = batch[1]

    cpu_device = torch.device("cpu")

    # We assume the data comes in via main memory. This is so we can time the effect
    #   of pushing it to gpu
    assert x.device == cpu_device

    model_device = model.device
    # Let's not penalize pushing stiminds, since the idea is that this is a static template
    # and therefore not streaming in in realtime.
    stiminds = stiminds.to(model_device)

    elapsed = []
    bidx = 0
    for _ in range(iterations):
        cur_x = x[bidx : bidx + 1, :, :]
        cur_stiminds = stiminds[bidx : bidx + 1, :, :]

        if model_device != cpu_device:
            elapsed.append(
                _time_inference_on_device(
                    model, session_id, cur_x, cur_stiminds, model_device
                )
            )
        else:
            elapsed.append(
                _time_inference_on_cpu(model, session_id, cur_x, cur_stiminds)
            )

        bidx = (bidx + 1) % dset_size

    return elapsed
