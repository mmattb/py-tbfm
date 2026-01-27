import numpy as np
import h5py
import pickle
import pandas as pd
import copy
import scipy.io
from scipy import signal, stats
import io


class Preprocess:
    def __init__(self):
        """ """
        self.data = None
        self.fs = None
        self.node_position = {}  # always starts at 0
        self.node_id = {}  # can be used to store original labeling
        self.time = None
        self.bad_channels = []  # 0 is reference index
        self.event_idx = {}
        self.metadata = {}

    def load_data(self):
        pass

    def get_data(self):
        pass

    def load_metadata(self):
        pass

    def remove_bad_channels(self):
        """
        list of bad channels. Numbering can start at 0 or 1 (default)
        :return: data with bad channels removed
        """
        if self.data is None:
            raise NotImplementedError("Load data first")
        elif self.node_position is None:
            raise NotImplementedError("Load node positions first")
        elif self.bad_channels is None:
            raise NotImplementedError("Load bad channels first")
        else:
            # remove bad channels
            # data = copy.deepcopy(self.data)
            for bad_ch in self.bad_channels[::-1]:
                self.data = np.delete(self.data, int(bad_ch), 0)

            # remove bad channels from electrode position dictionary
            node_pos = {}
            node_id = {}
            cnt = 0
            for i, key in enumerate(self.node_position):
                if key in self.bad_channels:
                    cnt += 1
                else:
                    node_pos[i - cnt] = self.node_position[key]
                    node_id[i - cnt] = self.node_id[key]

            self.node_position = node_pos
            self.node_id = node_id

            return self.data

    def bp_filter(self, fmin, fmax, order=3):
        """
        :param fmin: lower cutoff frequency
        :param fmax: upper cutoff frequency
        :param order: filter order (Butterworth filter)
        :return: filtered data
        """
        sos = signal.butter(
            N=order, Wn=[fmin, fmax], btype="bandpass", fs=self.fs, output="sos"
        )
        self.data = signal.sosfilt(sos, self.data)

        return self.data

    def remove_line_noise(self, f0=[60, 180, 300], Q=[20, 60, 100]):
        if self.fs is None:
            print("Specify sampling frequency first")
        else:
            for i in range(len(f0)):
                b, a = signal.iirnotch(f0[i], Q[i], fs=self.fs)
                self.data = signal.filtfilt(b, a, self.data)

        return self.data

    def remove_ulf(self, cutoff=0.1, N=2):
        sos = signal.butter(
            N=N,
            Wn=cutoff,
            btype="highpass",
            fs=self.fs,
            output="sos",
        )
        self.data = signal.sosfilt(sos, self.data, axis=1)
        return self.data

    def remove_artifacts(self, threshold):
        # identify artifacts
        data_norm = stats.zscore(self.data, axis=1)
        artifact_idx = np.where(abs(data_norm[0]) > threshold)[0]
        for ch in range(data_norm.shape[0] - 1):
            del_idx = []
            for k, a_idx in enumerate(artifact_idx):
                if abs(data_norm[ch + 1][a_idx]) <= threshold:
                    del_idx.append(k)
            artifact_idx = np.delete(artifact_idx, del_idx)

        # group artifact indicies
        artifact_idx_grouped = []
        current_group = []
        for idx in artifact_idx:
            if len(current_group) == 0:
                current_group.append(idx)
            elif current_group[-1] == idx - 1:
                current_group.append(idx)
            else:
                artifact_idx_grouped.append(current_group)
                current_group = [idx]
        if len(current_group) > 0:
            artifact_idx_grouped.append(current_group)

        # interpolate over artifacts
        idx_max = data_norm.shape[1] - 1
        for idx_group in artifact_idx_grouped:
            if (0 in idx_group) or (idx_max in idx_group):
                continue
            else:
                for ch in range(data_norm.shape[0]):
                    xp = [idx_group[0] - 1, idx_group[-1] + 1]
                    yp = self.data[ch][xp]
                    self.data[ch][idx_group] = np.interp(idx_group, xp, yp)

        return self.data

    def remove_mua(self, threshold):
        # identify artifacts
        data_norm = stats.zscore(self.data, axis=1)
        for ch in range(data_norm.shape[0]):
            artifact_idx = np.where(abs(data_norm[ch]) > threshold)[0]

            # group artifact indices
            artifact_idx_grouped = []
            current_group = []
            for idx in artifact_idx:
                if len(current_group) == 0:
                    current_group.append(idx)
                elif current_group[-1] == idx - 1:
                    current_group.append(idx)
                else:
                    artifact_idx_grouped.append(current_group)
                    current_group = [idx]
            if len(current_group) > 0:
                artifact_idx_grouped.append(current_group)

            # interpolate over artifacts
            idx_max = data_norm.shape[1] - 1
            for idx_group in artifact_idx_grouped:
                if (0 in idx_group) or (idx_max in idx_group):
                    continue
                else:
                    xp = [idx_group[0] - 1, idx_group[-1] + 1]
                    yp = self.data[ch][xp]
                    self.data[ch][idx_group] = np.interp(idx_group, xp, yp)

        return self.data


class OptoStimPreprocess(Preprocess):
    def __init__(
        self,
        session,
        base_path=None,
        load_exp_meta_data=True,
        bad_ch_file="bad_channels.pkl",
        tab_of_experiment="table_of_experiments.csv",
        electrode_pos="electrode_positions.pkl",
    ):
        """
        :param session: str
            experimental session. E.g. "MonkeyG_20150908_Session2_M1"
        :param base_path: str
            base path of data. LFP measurement can be in subdirectories
        """
        super().__init__()
        self.metadata["session"] = session
        self.metadata["base_path"] = base_path

        if load_exp_meta_data:
            self.load_metadata(bad_ch_file, tab_of_experiment, electrode_pos)

    def load_metadata(self, bad_ch_file, tab_of_experiment, electrode_pos):
        # load bad channels
        try:
            bad_channels = pickle.load(
                open(self.metadata["base_path"] + "/" + bad_ch_file, "rb")
            )
            self.bad_channels = [
                int(ch) - 1
                for ch in sorted(bad_channels[self.metadata["session"][:-3]])
            ]
        except:
            print(
                "File does not exist: " + self.metadata["base_path"] + "/" + bad_ch_file
            )

        # load experimental metadata
        try:
            table_of_exp = pd.read_csv(
                self.metadata["base_path"] + "/" + tab_of_experiment
            )
            idx = np.where(
                table_of_exp["File Name"] == self.metadata["session"] + ".zip"
            )[0][0]
            self.metadata.update(dict(table_of_exp.iloc[idx]))
            del self.metadata["Session"]
            if type(self.metadata["Delay"]) is str:
                self.metadata["Delay"] = int(self.metadata["Delay"].strip("ms"))
            self.metadata["m1_sites"] = [
                int(n) - 1 for n in self.metadata["m1_sites"].split(",")
            ]
            self.metadata["s1_sites"] = [
                int(n) - 1 for n in self.metadata["s1_sites"].split(",")
            ]
            self.metadata["stim_Coh_from"] -= 1
            self.metadata["stim_Coh_to"] -= 1
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + tab_of_experiment
            )

        # load electrode position
        try:
            with open(self.metadata["base_path"] + "/" + electrode_pos, "rb") as infile:
                electrode_pos = pickle.load(infile)
                for key in electrode_pos:
                    self.node_position[int(key) - 1] = electrode_pos[key]
                    self.node_id[int(key) - 1] = int(key)
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + electrode_pos
            )

    def load_data(self, block, file_extension=".mat", remove_mean=True, test_laser=1):
        """
        block: str
            measurement block. E.g. "CondBlock1" or "RecBlock1"
        """
        self.metadata["block"] = block

        if "Cond" in block or "Rec" in block:

            if "Cond" in block:
                sub_dir = "ConditioningBlocks"
            elif "Rec" in block:
                sub_dir = "RecordingBlocks"
            self.metadata["sub_dir"] = sub_dir

            file = (
                self.metadata["base_path"]
                + "/"
                + self.metadata["session"]
                + "/"
                + sub_dir
                + "/"
                + block
                + file_extension
            )
            self.metadata["file_location"] = file

            # load LFP data locally
            data = h5py.File(file, "r")
            keys = list(data.keys())
            keys.sort()
            i = 0
            make_signals = True
            for key in keys:
                if key.startswith("lfp"):
                    if make_signals:
                        signals = np.zeros((96, data[key].size))
                        make_signals = False
                    if remove_mean:
                        signals[i] = data[key][0] - np.mean(data[key][0])
                    else:
                        signals[i] = data[key][0]
                    i += 1

            self.data = signals

        elif "Test" in block:
            # TODO: revise this
            sub_dir = "TestingBlocks"
            self.metadata["sub_dir"] = sub_dir
            # lfp_matrix should be dictionary with 2 keys (2 lasers)
            # each entry consitst of tensor with ch x time x trial
            file = (
                self.metadata["base_path"]
                + "/"
                + self.metadata["session"]
                + "/"
                + sub_dir
                + "/"
                + block
                + file_extension
            )
            self.metadata["file_location"] = file
            self.metadata["test_laser"] = test_laser

            # load LFP data
            data = h5py.File(file, "r")
            keys = list(data.keys())
            keys.sort()
            make_signals1 = True
            make_signals2 = True
            i1 = 0
            for i, key in enumerate(keys):
                if key.startswith("lfp"):
                    if "traces" + str(test_laser) in key:
                        if make_signals1:
                            signals = np.zeros(
                                (96, data[key].shape[0], data[key].shape[1])
                            )
                            make_signals1 = False
                        signals[i1] = data[key][:]
                        i1 += 1

            self.data = signals

        # load metadata
        self.fs = data["samp_freq"][0][0]

        if "time" in keys:
            self.time = data["time"][0]
        if "stim1" in keys:
            self.event_idx["stim1"] = data["stim1"][0]
        if "stim2" in keys:
            self.event_idx["stim2"] = data["stim2"][0]
        if "tstart" in keys:
            self.metadata["t_start"] = data["tstart"][0][0]
        if "tend" in keys:
            self.metadata["t_end"] = data["tend"][0][0]
        if "win" in keys:
            self.metadata["test_block_rec_win"] = data["win"][:]

        return self.data

    def get_data(self, reference, L, start_idx=0, stim_idx=0, which_stim=1, offset=0):
        """
        :param reference: str
            'stim' or 'index'
        :param L:
        :param start_idx:
        :param stim_idx:
        :param which_stim:
        :param offset:
        :return:
        """
        if reference == "stim":
            if which_stim == 1:
                if not isinstance(self.event_idx["stim1"], np.ndarray):
                    raise IndexError("No stimulation by laser 1")
                else:
                    idx = np.where(self.time - self.event_idx["stim1"][stim_idx] >= 0)[
                        0
                    ][0]
                    # stim_time = self.event_idx['stim1'][stim_idx]
            elif which_stim == 2:
                if not isinstance(self.event_idx["stim2"], np.ndarray):
                    raise IndexError("No stimulation by laser 2")
                else:
                    idx = np.where(self.time - self.event_idx["stim2"][stim_idx] >= 0)[
                        0
                    ][0]
                    # stim_time = self.event_idx['stim2'][stim_idx]
            else:
                raise ValueError("which_stim has to be either 1 or 2")

            data_segment = self.data[:, idx + offset : idx + offset + L]

            # t = self.time[idx + offset:idx + offset + L]
            return data_segment

        elif reference == "idx":
            data_segment = self.data[:, start_idx + offset : start_idx + offset + L]
            return data_segment
        else:
            raise ValueError('Invalid reference. Must be either "stim" or "idx".')

    def update_channel_indices(self):
        # update stimulation electrodes
        if self.metadata["stim_Coh_from"] in self.bad_channels:
            self.metadata["stim_Coh_from"] = -2
        elif self.metadata["stim_Coh_from"] != -1:
            self.metadata["stim_Coh_from"] -= len(
                np.where(np.array(self.bad_channels) < self.metadata["stim_Coh_from"])[
                    0
                ]
            )

        if self.metadata["stim_Coh_to"] in self.bad_channels:
            self.metadata["stim_Coh_to"] = -2
        elif self.metadata["stim_Coh_to"] != -1:
            self.metadata["stim_Coh_to"] -= len(
                np.where(np.array(self.bad_channels) < self.metadata["stim_Coh_to"])[0]
            )

        # update M1 and S1 electrodes
        m1_sites = []
        for i, m1 in enumerate(self.metadata["m1_sites"]):
            if m1 not in self.bad_channels:
                m1_sites.append(m1 - len(np.where(np.array(self.bad_channels) < m1)[0]))
        self.metadata["m1_sites"] = m1_sites

        s1_sites = []
        for i, s1 in enumerate(self.metadata["s1_sites"]):
            if s1 not in self.bad_channels:
                s1_sites.append(s1 - len(np.where(np.array(self.bad_channels) < s1)[0]))
        self.metadata["s1_sites"] = s1_sites


class StrokePreprocess(Preprocess):
    def __init__(
        self,
        session,
        region,
        base_path=None,
        load_exp_meta_data=True,
        bad_ch_file="bad_channels.pkl",
        electrode_pos="electrode_positions.pkl",
    ):
        """
        :param session: 'PT4', 'PT5', 'PT6', 'PT7'
        :param region: 'ipsi', 'cont'
        :param base_path: default: 'E:/stroke_data/'
        :param load_exp_meta_data:
        :param bad_ch_file:
        :param electrode_pos:
        """
        super().__init__()
        self.metadata["session"] = session
        self.metadata["region"] = region
        self.metadata["base_path"] = base_path

        if load_exp_meta_data:
            self.load_metadata(bad_ch_file, electrode_pos)

    def load_metadata(self, bad_ch_file, electrode_pos):
        try:
            bad_channels = pickle.load(
                open(self.metadata["base_path"] + "/" + bad_ch_file, "rb")
            )
            self.bad_channels = [
                int(ch)
                for ch in sorted(
                    bad_channels[self.metadata["session"]][self.metadata["region"]]
                )
            ]
        except:
            print("File does not exist: " + self.base_path + "/" + bad_ch_file)

        try:
            with open(self.metadata["base_path"] + "/" + electrode_pos, "rb") as infile:
                electrode_pos = pickle.load(infile)
                for key in electrode_pos:
                    self.node_position[int(key)] = electrode_pos[key]
                    self.node_id[int(key)] = int(key)
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + electrode_pos
            )

        self.metadata["stim_node"] = 17
        self.fs = 1000.0

    def load_data(self, block, demean=True):
        """
        :param block:
            for PT5, PT6: 'baselineDS', 'duringPTDS', 'postPTDS', ('duringStimDS', 'postStimDS');
            PT4 has 'postPTDS1', 'postPTDS2', 'postPTDS';
            PT7 has: 'duringStimDS_1', 'duringStimDS_2' but no "postStimDS'
        :return: matrix containing local field potentials for all electrodes
        """
        file = (
            self.metadata["base_path"]
            + "/DownsampledSignals"
            + self.metadata["session"]
            + ".mat"
        )
        data = h5py.File(file, "r")
        key = block + "/" + self.metadata["region"]
        st = data[key]
        for i, ref in enumerate(st[:]):
            arr = data[ref[0]][0]
            if i == 0:
                self.data = np.zeros((32, len(arr)))
                self.data[0] = arr
            else:
                self.data[i] = arr

        self.time = np.arange(0, self.data.shape[1] * 1e-3, 1e-3)

        if demean:
            self.data = (self.data.T - np.mean(self.data, axis=1)).T

        return self.data

    def get_data(self, start_idx, L):
        data_segment = self.data[:, start_idx : start_idx + L]
        return data_segment

    def update_channel_indices(self):
        if self.metadata["stim_node"] in self.bad_channels:
            self.metadata["stim_node"] = -2
        else:
            self.metadata["stim_node"] -= len(
                np.where(np.array(self.bad_channels) < self.metadata["stim_node"])[0]
            )


class StrokeSplitPreprocess(Preprocess):
    def __init__(
        self,
        session,
        region,
        base_path=None,
        load_exp_meta_data=True,
        bad_ch_file="bad_channels.pkl",
        electrode_pos="electrode_positions.pkl",
    ):
        """
        :param session: 'PT4', 'PT5', 'PT6', 'PT7'
        :param region: 'ipsi', 'cont'
        :param base_path: default: 'E:/stroke_data/'
        :param load_exp_meta_data:
        :param bad_ch_file:
        :param electrode_pos:
        """
        super().__init__()
        self.metadata["session"] = session
        self.metadata["region"] = region
        self.metadata["base_path"] = base_path

        if load_exp_meta_data:
            self.load_metadata(bad_ch_file, electrode_pos)

    def load_metadata(self, bad_ch_file, electrode_pos):
        try:
            bad_channels = pickle.load(
                open(self.metadata["base_path"] + "/" + bad_ch_file, "rb")
            )
            self.bad_channels = [
                int(ch)
                for ch in sorted(
                    bad_channels[self.metadata["session"]][self.metadata["region"]]
                )
            ]
        except:
            print("File does not exist: " + self.base_path + "/" + bad_ch_file)

        try:
            with open(self.metadata["base_path"] + "/" + electrode_pos, "rb") as infile:
                electrode_pos = pickle.load(infile)
                for key in electrode_pos:
                    self.node_position[int(key)] = electrode_pos[key]
                    self.node_id[int(key)] = int(key)
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + electrode_pos
            )

        self.metadata["stim_node"] = 17
        self.fs = 1000.0

    def load_data(self, block_type, block, notch=False, demean=True):
        """
        :param block_type:
            PT7 has: 'baseG', 'postG', 'stimG'
        :param block: int
            typically 0-2 or 0-4.
        :return: matrix containing local field potentials for all electrodes
        """
        if notch:
            file = (
                self.metadata["base_path"]
                + "/SplitMonkey"
                + self.metadata["session"]
                + ".mat"
            )
        else:
            file = (
                self.metadata["base_path"]
                + "/NoNotchSplitMonkey"
                + self.metadata["session"]
                + ".mat"
            )
        data = scipy.io.loadmat(file)
        self.data = data[block_type][0][self.metadata["region"]][0][0][block].T
        if demean:
            self.data = (self.data.T - np.mean(self.data, axis=1)).T

        self.time = np.arange(0, self.data.shape[1] * 1e-3, 1e-3)

        return self.data

    def get_data(self, start_idx, L):
        data_segment = self.data[:, start_idx : start_idx + L]
        return data_segment

    def update_channel_indices(self):
        if self.metadata["stim_node"] in self.bad_channels:
            self.metadata["stim_node"] = -2
        else:
            self.metadata["stim_node"] -= len(
                np.where(np.array(self.bad_channels) < self.metadata["stim_node"])[0]
            )


class VertexPreprocess(Preprocess):
    def __init__(
        self,
        session,
        base_path="E:/vertex",
        load_exp_meta_data=True,
        electrode_pos=None,
        bad_ch=None,
    ):
        """
        :param base_path: default: 'E:/vertex/'
        :param load_exp_meta_data:
        :param electrode_pos:
        """
        super().__init__()
        self.metadata["base_path"] = base_path
        self.metadata["session"] = session

        if load_exp_meta_data:
            self.load_metadata(electrode_pos, bad_ch)

    def load_metadata(self, electrode_pos, bad_ch=None):
        try:
            with open(self.metadata["base_path"] + "/" + electrode_pos, "rb") as infile:
                electrode_pos = scipy.io.loadmat(infile)["coords"]
                for i in range(len(electrode_pos)):
                    self.node_position[i] = electrode_pos[i].astype(float)
                    self.node_id[i] = i
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + electrode_pos
            )

        if bad_ch is None:
            self.bad_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.bad_channels = bad_ch

        self.fs = 1000.0

    def load_data(self, demean=True, ignore_initialization=True):
        """
        :return: matrix containing local field potentials for all electrodes
        """
        file = self.metadata["base_path"] + "/" + self.metadata["session"] + ".mat"
        data = scipy.io.loadmat(file)
        self.data = data["lfp"]
        self.time = np.arange(0, self.data.shape[1] / self.fs, 1 / self.fs)

        if ignore_initialization:
            self.data = self.data[:, 250:-10]
            self.time = self.time[250:-10]

        if demean:
            self.data = (self.data.T - np.mean(self.data, axis=1)).T

        return self.data

    def get_data(self, start_idx, L):
        data_segment = self.data[:, start_idx : start_idx + L]
        return data_segment


class EightReachPreprocess(Preprocess):
    def __init__(
        self,
        monkey,
        date,
        site,
        base_path=None,
        load_exp_meta_data=True,
        bad_ch_file="bad_channels.pkl",
        tab_of_experiment="metadata.mat",
        electrode_pos="electrode_positions.pkl",
    ):
        """
        :param monkey: str
                "Jalapeno" or "GT"
        :param date: str
                date of experiment
        :param base_path: str
                base path of data. LFP measurement can be in subdirectories
        """
        super().__init__()
        self.metadata["monkey"] = monkey
        self.metadata["date"] = date
        self.metadata["site"] = site
        self.metadata["base_path"] = base_path

        if load_exp_meta_data:
            self.load_metadata(bad_ch_file, electrode_pos, tab_of_experiment)

    def load_metadata(self, bad_ch_file, electrode_pos, tab_of_experiment):
        # load bad channels
        try:
            file_path = (
                self.metadata["base_path"]
                + "/"
                + self.metadata["monkey"]
                + "/"
                + self.metadata["date"]
                + "/"
                + bad_ch_file
            )
            bad_channels = pickle.load(open(file_path, "rb"))
            bad_channels_idx_corrected = [
                int(ch) - 1 for ch in sorted(bad_channels[self.metadata["site"]])
            ]
            self.bad_channels = bad_channels_idx_corrected
        except:
            print("File does not exist: " + file_path)

        # load electrode positions
        try:
            with open(self.metadata["base_path"] + "/" + electrode_pos, "rb") as infile:
                electrode_pos = pickle.load(infile)
                for key in electrode_pos:
                    self.node_position[int(key)] = electrode_pos[key]
                    self.node_id[int(key)] = int(key)
        except:
            print(
                "File does not exist: " + self.metadata["base_path"] + "/" + bad_ch_file
            )

        self.fs = 3051.76

    def load_data(self, dir, trial, remove_mean=True, ds_factor=3):
        """
        Load all data for specified monkey, date, and array.
        Data is split by individual reach trials. Perform sampling rate conversion if desired

        :param dir: reach direction (integer from 1 to 8)
        :param trial: reach  trial (integer from 0 ... 24)
        :param remove_mean: if True, remove mean from each channel
        :param ds_factor: downsampling factor. If None use original sampling rate
        :return:
        """
        for ch in range(1, 97):
            data_path = (
                self.metadata["base_path"]
                + "/"
                + self.metadata["monkey"]
                + "/"
                + self.metadata["date"]
                + "/"
                + self.metadata["monkey"]
                + self.metadata["date"]
                + "_Reach8Dir_SeparateSegDir-1_"
                + self.metadata["site"]
                + "_Ch"
                + str(ch)
                + ".mat"
            )
            ch_data = scipy.io.loadmat(data_path)
            # initialize data matrices
            if ch == 1:
                if len(ch_data["ECoG"][0]["StartTarget"][0][dir - 1]) <= trial:
                    self.data = None
                    return None
                n_start = len(ch_data["ECoG"][0]["StartTarget"][0][dir - 1, trial])
                n_delay = len(ch_data["ECoG"][0]["InstructedDelay"][0][dir - 1, trial])
                n_reach = len(ch_data["ECoG"][0]["ReachTarget"][0][dir - 1, trial])
                if n_start < 10 or n_delay < 10 or n_reach < 10:
                    self.data = None
                    return None

                if type(ds_factor) is int:
                    if ds_factor > 1:
                        n = n_start + n_delay + n_reach
                        idxs = np.arange(0, n, 1)
                        n_start_new = len(np.where(idxs[::ds_factor] < n_start)[0])
                        n_delay_new = (
                            len(np.where(idxs[::ds_factor] < n_start + n_delay)[0])
                            - n_start_new
                        )
                        n_reach_new = len(idxs[::ds_factor]) - n_start_new - n_delay_new
                        n_start = n_start_new
                        n_delay = n_delay_new
                        n_reach = n_reach_new

                data = np.zeros((96, n_start + n_delay + n_reach))
                behavior_dct = {
                    "n_start": n_start,
                    "n_delay": n_delay,
                    "n_reach": n_reach,
                }

            # add ECoG and behavior data to dictionaries
            ecog_trial = np.concatenate(
                (
                    ch_data["ECoG"][0]["StartTarget"][0][dir - 1, trial].flatten(),
                    ch_data["ECoG"][0]["InstructedDelay"][0][dir - 1, trial].flatten(),
                    ch_data["ECoG"][0]["ReachTarget"][0][dir - 1, trial].flatten(),
                )
            )
            # downsample
            if type(ds_factor) is int:
                if len(ecog_trial) > ds_factor >= 1:
                    ecog_trial = signal.decimate(ecog_trial, q=ds_factor)
            # remove mean
            if remove_mean:
                data[ch - 1] = ecog_trial - np.mean(ecog_trial)
            else:
                data[ch - 1] = ecog_trial

            for key in [
                "TimeStart",
                "TimeEnd",
                "TrialNum",
                "ReachDelay",
                "OutcomeID",
                "AlwaysReward",
                "TrialDir",
                "TimeStartTarget",
                "TimeInstructedDelay",
                "TimeReach",
                "TickStartTarget",
                "TickInstructedDelay",
                "TickReach",
            ]:
                if len(ch_data["BahavioralData"][dir - 1, trial][0][key][0]) >= 1:
                    behavior_dct[key] = ch_data["BahavioralData"][dir - 1, trial][0][
                        key
                    ][0]
                else:
                    behavior_dct[key] = np.nan
            for key in [
                "Params",
                "PathStartTarget",
                "PathInstructedDelay",
                "PathReach",
                "OutcomeStr",
            ]:
                behavior_dct[key] = ch_data["BahavioralData"][dir - 1, trial][0][key][0]

        self.data = data
        self.behavior_dct = behavior_dct

        if type(ds_factor) is int:
            if ds_factor >= 1:
                self.fs /= ds_factor

        return data

    def load_all_data(self, remove_mean=True, ds_factor=3):
        """
        Load all data for specified monkey, date, and array.
        Data is split by individual reach trials. Perform sampling rate conversion if desired

        :param dir: reach direction (integer from 1 to 8)
        :param trial: reach  trial (integer from 0 ... 24)
        :param remove_mean: if True, remove mean from each channel
        :param ds_factor: downsampling factor. If None use original sampling rate
        :return:
        """
        ecog_dct = {}
        behavior_dct = {}
        for ch in range(1, 97):
            data_path = (
                self.metadata["base_path"]
                + "/"
                + self.metadata["monkey"]
                + "/"
                + self.metadata["date"]
                + "/"
                + self.metadata["monkey"]
                + self.metadata["date"]
                + "_Reach8Dir_SeparateSegDir-1_"
                + self.metadata["site"]
                + "_Ch"
                + str(ch)
                + ".mat"
            )
            ch_data = scipy.io.loadmat(data_path)
            # initialize data matrices
            if ch == 1:
                for dir in np.arange(1, 9, 1):
                    ecog_dct[dir] = {}
                    behavior_dct[dir] = {}
                    for trial in np.arange(0, 25, 1):
                        if len(ch_data["ECoG"][0]["StartTarget"][0][dir - 1]) <= trial:
                            continue
                        n_start = len(
                            ch_data["ECoG"][0]["StartTarget"][0][dir - 1, trial]
                        )
                        n_delay = len(
                            ch_data["ECoG"][0]["InstructedDelay"][0][dir - 1, trial]
                        )
                        n_reach = len(
                            ch_data["ECoG"][0]["ReachTarget"][0][dir - 1, trial]
                        )
                        if type(ds_factor) is int:
                            if ds_factor > 1:
                                n = n_start + n_delay + n_reach
                                idxs = np.arange(0, n, 1)
                                n_start_new = len(
                                    np.where(idxs[::ds_factor] < n_start)[0]
                                )
                                n_delay_new = (
                                    len(
                                        np.where(idxs[::ds_factor] < n_start + n_delay)[
                                            0
                                        ]
                                    )
                                    - n_start_new
                                )
                                n_reach_new = (
                                    len(idxs[::ds_factor]) - n_start_new - n_delay_new
                                )
                                n_start = n_start_new
                                n_delay = n_delay_new
                                n_reach = n_reach_new

                        ecog_dct[dir][trial] = np.zeros(
                            (96, n_start + n_delay + n_reach)
                        )
                        behavior_dct[dir][trial] = {
                            "n_start": n_start,
                            "n_delay": n_delay,
                            "n_reach": n_reach,
                        }

            # add ECoG and behavior data to dictionaries
            for dir in np.arange(1, 9, 1):
                for trial in np.arange(0, 25, 1):
                    if len(ch_data["ECoG"][0]["StartTarget"][0][dir - 1]) <= trial:
                        continue
                    ecog_trial = np.concatenate(
                        (
                            ch_data["ECoG"][0]["StartTarget"][0][
                                dir - 1, trial
                            ].flatten(),
                            ch_data["ECoG"][0]["InstructedDelay"][0][
                                dir - 1, trial
                            ].flatten(),
                            ch_data["ECoG"][0]["ReachTarget"][0][
                                dir - 1, trial
                            ].flatten(),
                        )
                    )
                    if len(ecog_trial) < 10:
                        continue
                    # downsample
                    if type(ds_factor) is int:
                        if ds_factor >= 1:
                            ecog_trial = signal.decimate(ecog_trial, q=ds_factor)
                    # remove mean
                    if remove_mean:
                        ecog_dct[dir][trial][ch - 1] = ecog_trial - np.mean(ecog_trial)
                    else:
                        ecog_dct[dir][trial][ch - 1] = ecog_trial

                    for key in [
                        "TimeStart",
                        "TimeEnd",
                        "TrialNum",
                        "ReachDelay",
                        "OutcomeID",
                        "AlwaysReward",
                        "TrialDir",
                        "TimeStartTarget",
                        "TimeInstructedDelay",
                        "TimeReach",
                        "TickStartTarget",
                        "TickInstructedDelay",
                        "TickReach",
                    ]:
                        if (
                            len(ch_data["BahavioralData"][dir - 1, trial][0][key][0])
                            >= 1
                        ):
                            behavior_dct[dir][trial][key] = ch_data["BahavioralData"][
                                dir - 1, trial
                            ][0][key][0]
                        else:
                            behavior_dct[dir][trial][key] = np.nan
                    for key in [
                        "Params",
                        "PathStartTarget",
                        "PathInstructedDelay",
                        "PathReach",
                        "OutcomeStr",
                    ]:
                        behavior_dct[dir][trial][key] = ch_data["BahavioralData"][
                            dir - 1, trial
                        ][0][key][0]

        self.ecog_dct = ecog_dct
        self.behavior_dct = behavior_dct

        if type(ds_factor) is int:
            if ds_factor >= 1:
                self.fs /= ds_factor

    def bp_filter_all_data(self, fmin, fmax, order=3):
        """
        :param fmin: lower cutoff frequency
        :param fmax: upper cutoff frequency
        :param order: filter order (Butterworth filter)
        :return: filtered data
        """
        sos = signal.butter(
            N=order, Wn=[fmin, fmax], btype="bandpass", fs=self.fs, output="sos"
        )
        for dir in np.arange(1, 9, 1):
            for trial in np.arange(0, 25, 1):
                self.ecog_dct[dir][trial] = signal.sosfilt(
                    sos, self.ecog_dct[dir][trial]
                )

        return self.ecog_dct

    def remove_bad_channels_from_dct(self):
        """
        list of bad channels. Numbering can start at 0 or 1 (default)
        :return: data with bad channels removed
        """
        # remove bad channels
        # data = copy.deepcopy(self.data)
        for dir in np.arange(1, 9, 1):
            for trial in self.ecog_dct[dir]:
                for bad_ch in self.bad_channels[::-1]:
                    self.ecog_dct[dir][trial] = np.delete(
                        self.ecog_dct[dir][trial], int(bad_ch), 0
                    )

        # remove bad channels from electrode position dictionary
        node_pos = {}
        node_id = {}
        cnt = 0
        for i, key in enumerate(self.node_position):
            if key in self.bad_channels:
                cnt += 1
            else:
                node_pos[i - cnt] = self.node_position[key]
                node_id[i - cnt] = self.node_id[key]

        self.node_position = node_pos
        self.node_id = node_id

        return self.ecog_dct


# class for paired electrical Stimulation experiment
class PairedEStimPreprocess(Preprocess):
    def __init__(
        self,
        session,
        base_path=None,
        load_exp_meta_data=True,
        electrode_pos="electrode_positions.pkl",
    ):
        """
        :param session: str
            experimental session. E.g. "20150911"
        :param base_path: str
            base path of data. LFP measurement can be in subdirectories
        """
        super().__init__()
        self.metadata["session"] = session
        self.metadata["base_path"] = base_path

        if load_exp_meta_data:
            self.load_metadata(electrode_pos)

    def load_metadata(self, electrode_pos):
        # load bad channels
        self.bad_channels = []

        # load experimental metadata
        self.metadata["block_type"] = {}
        self.metadata["stim_location"] = {}
        if self.metadata["session"] == "20150911":
            self.metadata["good_blocks"] = [
                1,
                2,
                3,
                4,
                5,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
            ]
            for block in self.metadata["good_blocks"]:
                if block in [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
                    self.metadata["block_type"][block] = "control"
                    self.metadata["stim_location"][block] = None
                elif block in [7, 8, 9, 10]:
                    self.metadata["block_type"][block] = "in_stdp"
                    self.metadata["stim_location"][block] = [2, 80]
                elif block in [16, 17, 18, 19, 20]:
                    self.metadata["block_type"][block] = "out_stdp"
                    self.metadata["stim_location"][block] = [2, 80]

        elif self.metadata["session"] == "20150914":
            self.metadata["good_blocks"] = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ]
            for block in self.metadata["good_blocks"]:
                if block in [1, 2, 3, 9, 10, 11, 17, 18, 19]:
                    self.metadata["block_type"][block] = "control"
                    self.metadata["stim_location"][block] = None
                elif block in [12, 13, 14, 15, 16]:
                    self.metadata["block_type"][block] = "in_stdp"
                    self.metadata["stim_location"][block] = [1, 79]
                elif block in [4, 5, 6, 7, 8]:
                    self.metadata["block_type"][block] = "out_stdp"
                    self.metadata["stim_location"][block] = [1, 79]

        elif self.metadata["session"] == "20150916":
            self.metadata["good_blocks"] = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ]
            for block in self.metadata["good_blocks"]:
                if block in [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 20, 21, 25, 26]:
                    self.metadata["block_type"][block] = "control"
                    self.metadata["stim_location"][block] = None
                elif block in [17, 18, 19, 22, 23, 24]:
                    self.metadata["block_type"][block] = "in_stdp"
                    if block in [17, 18, 19]:
                        self.metadata["stim_location"][block] = [92, 47]
                    else:
                        self.metadata["stim_location"][block] = [7, 75]
                elif block in [9, 10, 11, 12, 13]:
                    self.metadata["block_type"][block] = "out_stdp"
                    self.metadata["stim_location"][block] = [1, 79]

        elif self.metadata["session"] == "20150918":
            self.metadata["good_blocks"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            for block in self.metadata["good_blocks"]:
                if block in [1, 2, 3, 9, 10, 11]:
                    self.metadata["block_type"][block] = "control"
                    self.metadata["stim_location"][block] = None
                elif block in [4, 5, 6, 7, 8]:
                    self.metadata["block_type"][block] = "in_stdp"
                    self.metadata["stim_location"][block] = [79]

        # load electrode positions
        try:
            with open(electrode_pos, "rb") as infile:
                electrode_pos = pickle.load(infile)
                for key in electrode_pos:
                    self.node_position[int(key) - 1] = electrode_pos[key]
                    self.node_id[int(key) - 1] = int(key)
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + electrode_pos
            )

        self.fs = 3051.0

    # load data for specified block
    def load_data(self, block, remove_mean=True, ds_factor=3):
        """
        :param block: int
            block number
        :param remove_mean: bool
            if True, remove mean from each channel
        :param ds_factor: int
            downsampling factor. If None use original sampling rate
        :return: matrix containing local field potentials for all electrodes
        """

        # adjust sampling rate if ds_factor >= 1
        if type(ds_factor) is int:
            if ds_factor >= 1:
                self.fs /= ds_factor

        file = (
            self.metadata["base_path"]
            + "/"
            + self.metadata["session"]
            + "/processed/lfp_"
            + str(block).zfill(2)
            + ".mat"
        )
        f = scipy.io.loadmat(file)
        keys = list(f.keys())
        keys.sort()
        i = 0
        make_signals = True
        for key in keys:
            if key.startswith("ad_samp"):
                if make_signals:
                    if type(ds_factor) is int:
                        if ds_factor >= 1:
                            T = signal.decimate(f[key][:, 0], q=ds_factor).size
                            self.data = np.zeros((96, T))
                        else:
                            self.data = np.zeros((96, f[key].size))
                    make_signals = False
                # downsample signal
                if type(ds_factor) is int:
                    if ds_factor >= 1:
                        self.data[i] = signal.decimate(f[key][:, 0], q=ds_factor)
                else:
                    self.data[i] = f[key][:, 0]
                i += 1
        self.time = np.arange(0, self.data.shape[1] / self.fs, 1 / self.fs)

        if remove_mean:
            self.data = (self.data.T - np.mean(self.data, axis=1)).T

        return self.data

    def get_data(self, start_idx, L):
        data_segment = self.data[:, start_idx : start_idx + L]
        return data_segment


class LaserInhibitPreprocess(Preprocess):
    def __init__(
        self,
        session,
        block_type,
        base_path=None,
        load_exp_meta_data=True,
        bad_ch_file="bad_channels.pkl",
        electrode_pos="electrode_positions.pkl",
    ):
        """
        :param session: 'L_Session3', 'L_Session5', 'L_Session7', 'H_Session2', 'H_Session3', 'H_Session4',
                            'H_Session6', 'H_Session7'
        :param block_type: 'baseline', 'stim'
        :param block: int: 0, ..., 5 for 'baseline', 1, ..., 5 for 'stim'
        :param base_path: default: None
        :param load_exp_meta_data:
        :param bad_ch_file:
        :param electrode_pos:
        """
        super().__init__()
        self.metadata["session"] = session
        self.metadata["block_type"] = block_type
        self.metadata["base_path"] = base_path
        self.base_path = base_path

        if load_exp_meta_data:
            self.load_metadata(bad_ch_file, electrode_pos)

    def load_metadata(self, bad_ch_file, electrode_pos):
        try:
            bad_channels = pickle.load(
                open(self.metadata["base_path"] + "/" + bad_ch_file, "rb")
            )
            self.bad_channels = [
                int(ch) for ch in sorted(bad_channels[self.metadata["session"]])
            ]
        except:
            print("File does not exist: " + self.base_path + "/" + bad_ch_file)

        try:
            with open(self.metadata["base_path"] + "/" + electrode_pos, "rb") as infile:
                electrode_pos = pickle.load(infile)
                for key in electrode_pos:
                    self.node_position[int(key)] = electrode_pos[key]
                    self.node_id[int(key)] = int(key)
        except:
            print(
                "File does not exist: "
                + self.metadata["base_path"]
                + "/"
                + electrode_pos
            )

        # set response and control sessions
        self.metadata["rsp_sessions"] = [
            "L_Session5",
            "L_Session7",
            "H_Session2",
            "H_Session4",
            "H_Session7",
        ]
        self.metadata["ctl_sessions"] = ["L_Session3", "H_Session3", "H_Session6"]

        self.fs = 1000.0

    def load_data(self, block, demean=True):
        """
        :param block: int: 0, ..., 5 for 'baseline', 1, ..., 5 for 'stim'
        :return: matrix containing local field potentials for all electrodes
        """
        if self.metadata["session"] in self.metadata["rsp_sessions"]:
            with open(
                self.metadata["base_path"] + "laser_inhibition_response_data.pkl", "rb"
            ) as f:
                data_file = pickle.load(f)
        elif self.metadata["session"] in self.metadata["ctl_sessions"]:
            with open(
                self.metadata["base_path"] + "laser_inhibition_control_data.pkl", "rb"
            ) as f:
                data_file = pickle.load(f)
        else:
            raise ValueError("Invalid session. Cannot load data")

        self.data = data_file[self.metadata["session"]][self.metadata["block_type"]][
            str(block)
        ].T
        self.time = np.arange(0, self.data.shape[1] * 1 / self.fs, 1 / self.fs)

        # load stim channel
        stim_ch_str = data_file[self.metadata["session"]]["stim_ch"]
        self.metadata["stim_node"] = (
            int(stim_ch_str[1:]) - 1
            if stim_ch_str[0] == "a"
            else int(stim_ch_str[1:]) + 15
        )

        if demean:
            self.data = (self.data.T - np.mean(self.data, axis=1)).T

        return self.data

    def get_data(self, start_idx, L):
        data_segment = self.data[:, start_idx : start_idx + L]
        return data_segment

    def update_channel_indices(self):
        if self.metadata["stim_node"] in self.bad_channels:
            self.metadata["stim_node"] = -2
        else:
            self.metadata["stim_node"] -= len(
                np.where(np.array(self.bad_channels) < self.metadata["stim_node"])[0]
            )
