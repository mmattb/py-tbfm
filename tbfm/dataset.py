"""
Data sets and samplers.
"""

from collections import namedtuple
import copy
import os
import pickle
import random

import torch

import pandas as pd

from torch.utils.data import Dataset, DataLoader, IterableDataset

# Default device we load data onto. None means main memory.
DEVICE = None

# Metadata for a single session
Meta = namedtuple(
    "Meta", "session_id stim_coh_from stim_coh_to delay interpair_interval"
)


def get_clock(timelen, dtype=torch.float, device=None):
    return torch.arange(timelen, dtype=dtype, device=device) / timelen


def load_meta(root_dir="data", sessions_to_keep=None):
    """
    Load metadata from experiment table.
    Args:
      root_dir (str): the path to the root data directory. If it's relative,
            must be relative to the running process. Default assumes code is
            running from the repo root.
      sessions_to_keep: collection[str] containing session_ids we want to keep.
            Preferably something O(1) in value lookup.

    Returns: list[Meta]
    """
    infile = os.path.join(root_dir, "table_of_experiments.csv")
    metaf = pd.read_csv(infile)
    cond_experiments = metaf.loc[metaf["Number of Lasers during Conditioning"] == 2]

    meta_out = []

    for _, c in cond_experiments.iterrows():
        sid = c["File Name"][:-4]
        if sessions_to_keep and sid not in sessions_to_keep:
            continue
        stim_coh_from = int(c["stim_Coh_from"])
        stim_coh_to = int(c["stim_Coh_to"])
        delay = int(c["Delay"][:-2])
        interpair_interval = int(c["stim1-to-stim1 interval (msecs)"])

        if stim_coh_from == 0 or stim_coh_to == 0:
            continue

        meta_out.append(
            Meta(sid, stim_coh_from, stim_coh_to, delay, interpair_interval)
        )

    return meta_out


class SessionInMemoryDataset(Dataset):
    """
    A dataset for a single session.
    """

    def __init__(
        self,
        torchdir,
        window_size,
        runway,
        include_clockvec=True,
        unpack_stiminds=False,
        device=DEVICE,
    ):
        """
        torchdir: path to the pytorch data for the given session.
        This implementation holds the whole set in memory. Take care!
        
        Args:
            torchdir: path to the pytorch data for the given session
            window_size: size of the window
            runway: runway length
            include_clockvec: if True, include clock vector in stim_ind
            unpack_stiminds: if True, instead of stiminds of (batch, trial_len, 3),
                            use (batch, 2) with normalized pulse positions. Ignores
                            clock vector.
            device: device to load data onto
        """
        self._device = device
        self._torchdir = torchdir
        self.include_clockvec = include_clockvec
        self.unpack_stiminds = unpack_stiminds
        self._runway = runway

        self._chunk_paths = [
            de for de in os.listdir(torchdir) if de.startswith("cond.")
        ]
        self._chunk_count = len(self._chunk_paths)
        self._meta = pickle.load(open(os.path.join(torchdir, "meta.pkl"), "rb"))

        self._pulse_count = self._meta["num_pulses"]

        some_chunk = torch.load(
            os.path.join(torchdir, "cond.0.torch"),
            map_location="cpu",
            weights_only=False,
        )
        self._num_feats = some_chunk.shape[1]
        self._pwinlen = some_chunk.shape[2]
        self._dtype = some_chunk.dtype
        del some_chunk

        if window_size is None:
            window_size = self._pwinlen

        self.x = torch.zeros(
            self._pulse_count,
            window_size,
            self._num_feats,
            dtype=self._dtype,
            device=device,
        )
        
        # Stim_ind shape depends on unpack_stiminds flag
        if unpack_stiminds:
            # Unpacked: (batch, 2) with normalized pulse positions only
            # No clock dimension needed
            self.stim_ind = torch.zeros(
                self._pulse_count,
                2,
                dtype=self._dtype,
                device=device,
            )
        else:
            # Original: (batch, trial_len, stimdim)
            self.stim_ind = torch.zeros(
                self._pulse_count,
                window_size,
                3 if include_clockvec else 2,
                dtype=self._dtype,
                device=device,
            )

        # Here we gooooo...
        self.x = self.x.pin_memory()
        self.stim_ind = self.stim_ind.pin_memory()
        cur_batch_loc = 0
        for cidx in range(self._chunk_count):
            cur_x, cur_i = self._load_chunk(cidx)
            bsize = cur_x.shape[0]

            self.x[cur_batch_loc : cur_batch_loc + bsize, :, :].copy_(
                cur_x[:, :window_size, :]
            )
            self.stim_ind[cur_batch_loc : cur_batch_loc + bsize].copy_(cur_i)

            cur_batch_loc += bsize

        self._window_size = window_size
        # Our batch items are going to be windows cut from each pulse window
        self._timelen = window_size
        self._w_per_pulse = self._pwinlen - window_size
        self._len = self._pulse_count * self._w_per_pulse
        self._batch_size = None

    def _load_chunk(self, cidx):
        cond = torch.load(
            os.path.join(self._torchdir, f"cond.{cidx}.torch"),
            map_location=self._device,
            weights_only=False,
        )
        pulses = torch.load(
            os.path.join(self._torchdir, f"pulse_loc.{cidx}.torch"),
            map_location=self._device,
            weights_only=False,
        )

        # Supplement feats with stim locs
        if self.unpack_stiminds:
            # Unpacked format: (batch, 2) with normalized pulse positions only
            stim_ind = torch.zeros(
                cond.shape[0], 2, device=self._device, dtype=self._dtype
            )

            window_size = cond.shape[2]
            
            for idx, pulse in enumerate(pulses):
                assert len(pulse) == 3
                # Normalize pulse positions to [0, 1]
                stim_ind[idx, 0] = pulse[0] / window_size
                stim_ind[idx, 1] = pulse[1] / window_size
        else:
            # Original format: (batch, trial_len, stimdim) with 1-hot encoding
            stim_dim = 3 if self.include_clockvec else 2
            stim_ind = torch.zeros(
                cond.shape[0], cond.shape[2] - 1, stim_dim, device=self._device
            )
            clock_vec = get_clock(stim_ind.shape[1], dtype=self._dtype, device=self._device)
            
            for idx, pulse in enumerate(pulses):
                assert len(pulse) == 3
                stim_ind[idx, pulse[0], 0] = 1
                stim_ind[idx, pulse[1], 1] = 1

                if self.include_clockvec:
                    stim_ind[idx, :, 2] = clock_vec

        x = cond[:, :, :].permute(0, 2, 1)

        return x, stim_ind

    def test_split(self, pct_or_count, pct_or_count_test=None):
        class Subset:
            def __init__(self, outer, runway, start_idx, end_idx):
                """ """
                self._outer = outer
                self._start_idx = start_idx
                self._end_idx = end_idx
                self._runway = runway

                self._len = end_idx - start_idx

                self._cur_idx = None
                self._batch_size = None

            def __len__(self):
                return self._len

            def test_split(self, pct_or_count, pct_or_count_test=None):
                if isinstance(pct_or_count, float):
                    split = int(pct_or_count * self._len)
                else:
                    split = pct_or_count

                start_idx = self._start_idx
                split_idx = split_idx_test = start_idx + split

                if pct_or_count_test is not None:
                    if isinstance(pct_or_count_test, float):
                        split_test = int(pct_or_count_test * self._len)
                    else:
                        split_test = pct_or_count_test
                    split_idx_test = self._end_idx - split_test

                end_idx = self._end_idx

                train_set = Subset(self._outer, self.runway, start_idx, split_idx)
                test_set = Subset(self._outer, self.runway, split_idx_test, end_idx)

                return train_set, test_set

            @property
            def dtype(self):
                return self._outer.dtype

            def __getitem__(self, idx):
                bidx = self._start_idx + idx
                if bidx > len(self):
                    return self[bidx - len(self)]
                return self._outer[bidx]

            def set_batch_iter(self, batch_size):
                self._cur_idx = self._start_idx
                self._batch_size = batch_size

            @property
            def runway(self):
                return self._runway

            @property
            def num_feats(self):
                return self._outer.num_feats

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur_idx >= self._end_idx:
                    raise StopIteration()

                start_idx = self._cur_idx
                end_idx = min(self._end_idx, start_idx + self._batch_size)
                runway = self._outer.x[start_idx:end_idx, : self.runway, :]
                
                # Handle unpacked vs packed stim_ind
                if self._outer.unpack_stiminds:
                    # Unpacked: (batch, stimdim) - no temporal slicing needed
                    stim_ind = self._outer.stim_ind[start_idx:end_idx, :]
                else:
                    # Packed: (batch, trial_len, stimdim) - slice after runway
                    stim_ind = self._outer.stim_ind[start_idx:end_idx, self.runway :, :]
                
                y = self._outer.x[start_idx:end_idx, self.runway :, :]
                self._cur_idx += self._batch_size

                return runway, stim_ind, y

        if isinstance(pct_or_count, float):
            split = int(pct_or_count * self._len)
        else:
            split = pct_or_count

        start_idx = 0
        split_idx = split_idx_test = start_idx + split

        if pct_or_count_test is not None:
            if isinstance(pct_or_count_test, float):
                split_test = int(pct_or_count_test * len(self))
            else:
                split_test = pct_or_count_test
            split_idx_test = len(self) - split_test

        train_set = Subset(self, self.runway, start_idx, split_idx)
        test_set = Subset(self, self.runway, split_idx_test, len(self))

        return train_set, test_set

    @property
    def pulse_count(self):
        return self._pulse_count

    @property
    def windows_per_pulse(self):
        return self._w_per_pulse

    @property
    def window_size(self):
        return self._window_size

    @property
    def timelen(self):
        return self._timelen

    @property
    def pwinlen(self):
        return self._pwinlen

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_feats(self):
        return self._num_feats

    @property
    def runway(self):
        return self._runway

    def set_batch_iter(self, batch_size):
        self._cur_idx = 0
        self._batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._cur_idx >= len(self):
            raise StopIteration()

        start_idx = self._cur_idx
        end_idx = min(len(self), start_idx + self._batch_size)
        runway = self.x[start_idx:end_idx, : self.runway, :]
        stim_ind = self.stim_ind[start_idx:end_idx, self.runway :, :]
        y = self.x[start_idx:end_idx, self.runway :, :]

        self._cur_idx += self._batch_size

        return runway, stim_ind, y

    def __len__(self):
        return self._len

    def _get_item_window(self, bidx, tstart, tend):
        # bidx is batch index; i.e. the pulse id
        x = self.x[bidx, tstart:tend, :]
        stim_ind = self.stim_ind[bidx, tstart:tend, :]
        runway = x[: self.runway, :]
        y = x[self.runway :, :]

        return runway, stim_ind, y

    def __getitem__(self, widx):
        # bidx is batch index; i.e. the pulse id
        bidx = widx // self._w_per_pulse
        tstart = widx % self._w_per_pulse
        tend = tstart + self.timelen
        return self._get_item_window(bidx, tstart, tend)


class InfiniteDsetIter:
    def __init__(self, session_dset):
        self.dset = session_dset

    def get(self):
        while True:
            it = iter(self.dset)
            while True:
                try:
                    output = next(it)
                    yield output
                except StopIteration:
                    break


class SessionLoader:
    """
    Provide batches containing pulse pairs from across sessions. Tries to
    ensure we get close to the same number of examples from each session without
    exceeding batches of batch_size. We may not get batches exactly batch_size.
    NOTE: not quite the same as a pytorch DataLoader, but similar idea.
    """

    def __init__(
        self,
        meta,
        subdir,
        runway,
        session_subdir="torch",
        batch_size=1000,
        window_size=None,
        in_memory=False,
        unpack_stiminds=False,
        device=DEVICE,
    ):
        """
        Args:
            meta: Iterable[Meta] for each session we are including
            subdir: path to data directory; e.g. "./data"
            batch_size: [int]: our batches will be <= this in size
            unpack_stiminds: if True, use unpacked (batch, stimdim) format for stim_ind
            device: something passable to tensor.to()
        """
        self._meta = meta
        self._device = device
        self._runway = runway
        self.unpack_stiminds = unpack_stiminds

        num_sessions = len(meta)
        self._batch_size_per_session = batch_size // num_sessions

        # {session id: (torch metadata, path to torch data, dataloader)}
        self._access_paths = {}

        # dloader iterables: {session_id: iterable}
        self._iters = {}

        # {session_id: dict}
        self._torchmeta = {}

        # Session ids of iterators we have reset. Once all have been reset at least
        #  once, the self iterator has been exhausted.
        self._reset_iters = set()

        self._dtype = None
        self._timelen = None
        for m in meta:
            curdir = os.path.join(subdir, m.session_id)
            torchdir = os.path.join(curdir, session_subdir)
            torch_meta = pickle.load(open(os.path.join(torchdir, "meta.pkl"), "rb"))
            self._torchmeta[m.session_id] = torch_meta

            if in_memory:
                dset = SessionInMemoryDataset(
                    torchdir, 
                    runway=runway, 
                    window_size=window_size, 
                    unpack_stiminds=unpack_stiminds,
                    device=device
                )
            else:
                # dset = SessionDataset(torchdir, window_size=window_size, device=device)
                raise NotImplementedError()

            self._dtype = dset.dtype
            self._timelen = dset.timelen

            self._access_paths[m.session_id] = (torch_meta, torchdir, dset)

        if self._dtype is None:
            raise ValueError("Error initializing; did we find no data to load?")

    def train_test_split(self, train_cut, test_cut=None):
        """
        Returns a pair of SessionLoaders. Train cut is pct or count.
        """
        trainloader = copy.copy(self)
        testloader = copy.copy(self)

        trainloader._iters = {}
        trainloader._reset_iters = set()
        testloader._iters = {}
        testloader._reset_iters = set()

        trainloader._access_paths = {}
        testloader._access_paths = {}
        for session_id, ap in self._access_paths.items():
            torch_meta, torchdir, dset = ap

            trainset, testset = dset.test_split(train_cut, pct_or_count_test=test_cut)

            trainloader._access_paths[session_id] = (torch_meta, torchdir, trainset)
            testloader._access_paths[session_id] = (torch_meta, torchdir, testset)

        return trainloader, testloader

    def get_first_dset(self):
        session_id = list(self._access_paths.keys())[0]
        return self.get_dset(session_id)

    def get_batch_for_ms_graph(self, batch_size):
        """
        Rotate out one session's data for multistep loss graph.
        This will not be windowed.
        """
        session_id = random.choice(list(self._access_paths.keys()))
        dset = self.get_dset(session_id)

        if batch_size is None:
            batch_size = len(dset.dataset)

        # i.e. 1 window of size equal to the whole trial time minus 1 (for the x vs y lag).
        dset = dset.dataset.rewindow(1, dset.dataset.pwinlen - 1)
        dataloader = DataLoader(dset, batch_size=batch_size)
        return session_id, next(iter(dataloader))

    def get_dset(self, session_id):
        """
        Get the underlying dataset for a given session.
        Args:
          * session_id [str]
        """
        dset = self._access_paths[session_id][2]
        return dset

    @property
    def dtype(self):
        """
        Pytorch data type of the dataset
        """
        return self._dtype

    @property
    def timelen(self):
        """
        The number of time steps of each batch element
        """
        return self._timelen

    @property
    def meta(self):
        """
        Metadata for the sesions' data.
        """
        return copy.copy(self._meta)

    @property
    def torchmeta(self):
        """
        Note we have two metadata sources: the sessions table, and a pytorch-only
        meta which provides details of the tensors we split out.
        """
        return copy.copy(self._torchmeta)

    def __next__(self):
        """
        Returns:
           Dict {session_id: (batch_size, x, stimind)}
        """
        data = {}

        # dloader iterables: {session_id: iterable}
        new_iters = {}
        for session_id, it in self._iters.items():
            # Will naturally bubble up StopIteration once chunks are exhausted.
            try:
                new_runway, new_stim, new_y = next(it)
            except StopIteration:
                if len(self._reset_iters) == (len(self._iters) - 1):
                    raise

                it = iter(self._access_paths[session_id][2])
                new_runway, new_stim, new_y = next(it)
                self._reset_iters.add(session_id)

            new_iters[session_id] = it

            data[session_id] = (new_runway, new_stim, new_y)

        self._iters = new_iters

        return data

    def __iter__(self):
        self._iters = {}
        for k, apv in self._access_paths.items():
            dset = apv[2]
            dset.set_batch_iter(self._batch_size_per_session)
            self._iters[k] = iter(dset)
        self._reset_iters = set()
        return self

    @property
    def num_sessions(self):
        """
        Number of sessions this Loader tracks.
        """
        return len(self._access_paths)

    @property
    def session_ids(self):
        """
        Session IDs in this loader/set
        """
        return list(self._access_paths.keys())

    def keys(self):
        return self.session_ids

    def get_session_num_feats(self, session_id):
        return self._access_paths[session_id][2].num_feats


def load_data_some_sessions(
    paths,
    runway=20,
    subdir="data",
    session_subdir="torch",
    batch_size=1000,
    window_size=None,
    in_memory=True,
    unpack_stiminds=False,
    device=DEVICE,
):
    """
    Provide a SessionLoader for a subset of sessions.
    Args:
        paths: Iterable[str] - paths to per-session dirs
        subdir: str - data directory
        batch_size: [int]: our batches will be <= this in size
        unpack_stiminds: if True, use unpacked (batch, 2) format for stim_ind
        device: something passable to tensor.to()
    """
    sessions_to_keep = {os.path.basename(fqp) for fqp in paths}
    meta = load_meta(root_dir=subdir, sessions_to_keep=sessions_to_keep)
    trainloader = SessionLoader(
        meta,
        subdir,
        runway=runway,
        session_subdir=session_subdir,
        batch_size=batch_size,
        window_size=window_size,
        in_memory=in_memory,
        unpack_stiminds=unpack_stiminds,
        device=device,
    )
    return trainloader


def load_data_all_sessions(
    runway=20,
    subdir="data",
    num_held_out_sessions=10,
    held_out_session_ids=None,
    session_subdir="torch",
    batch_size=1000,
    in_memory=True,
    window_size=None,
    unpack_stiminds=False,
    device=DEVICE,
):
    """
    Provide a SessionLoader for all sessions, less some holdouts.
    Args:
        subdir: str - data directory
        num_held_out_sessions: int - number of sessions to hold out from the training loader
        held_out_session_ids: Iterable[str] - a specific set of session_ids for sessions to hold out
        batch_size: [int]: our batches will be <= this in size
        unpack_stiminds: if True, use unpacked (batch, 2) format for stim_ind
        device: something passable to tensor.to()

    Returns:
        SessionLoader (for training), Iterable[str] (session_ids for held out sessions)
    """
    if (
        held_out_session_ids
        and num_held_out_sessions
        and len(held_out_session_ids) != num_held_out_sessions
    ):
        raise ValueError("Must align num_held_out_sessions, held_out_session_ids")

    meta = load_meta(subdir)

    if held_out_session_ids:
        train_sessions = []
        held_out_sessions = []
        for m in meta:
            if m.session_id in held_out_session_ids:
                held_out_sessions.append(m)
            else:
                train_sessions.append(m)
    else:
        random.shuffle(meta)
        train_cut = -1 * num_held_out_sessions

        train_sessions = meta[:train_cut]
        held_out_sessions = meta[train_cut:]

    trainloader = SessionLoader(
        train_sessions,
        subdir,
        runway=runway,
        session_subdir=session_subdir,
        window_size=window_size,
        in_memory=in_memory,
        batch_size=batch_size,
        unpack_stiminds=unpack_stiminds,
        device=device,
    )

    return trainloader, held_out_sessions
