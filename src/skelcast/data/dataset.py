import os
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .prepare_data import (
    get_skeleton_files,
    get_missing_files,
    filter_missing,
    should_blacklist,
)

from skelcast.data import DATASETS, COLLATE_FUNCS

def read_skeleton_file(
    file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True
) -> dict:
    """
    Copied from https://github.com/shahroudy/NTURGB-D/blob/master/Python/txt2npy.py
    """
    f = open(file_path, "r")
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat["file_name"] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat["nbodys"] = []
    bodymat["njoints"] = njoints
    for body in range(max_body):
        if save_skelxyz:
            bodymat["skel_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat["rgb_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat["depth_body{}".format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            continue
        # skip the empty frame
        bodymat["nbodys"].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = "skel_body{}".format(body)
            rgb_body = "rgb_body{}".format(body)
            depth_body = "depth_body{}".format(body)

            bodyinfo = datas[cursor][:-1].split(" ")
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(" ")
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame, joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame, joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame, joint] = jointinfo[5:7]
    # prune the abundant bodys
    for each in range(max_body):
        if each >= max(bodymat["nbodys"]):
            if save_skelxyz:
                del bodymat["skel_body{}".format(each)]
            if save_rgbxy:
                del bodymat["rgb_body{}".format(each)]
            if save_depthxy:
                del bodymat["depth_body{}".format(each)]
    return bodymat

@dataclass
class NTURGBDSample:
    x: torch.tensor
    y: torch.tensor
    label: Tuple[int, str]
    mask: Optional[torch.tensor] = None

def nturbgd_collate_fn_with_overlapping_context_window(batch: List[NTURGBDSample]) -> NTURGBDSample:
    # TODO: Normalize each sample individually along its 3 axes 
    batch_x = torch.cat([item.x for item in batch], dim=0)
    batch_y = torch.cat([item.y for item in batch], dim=0)
    batch_label = [item.label for item in batch]

    # batch_x = default_collate(batch_x)
    # batch_y = default_collate(batch_y)
    batch_label = default_collate(batch_label)
    return NTURGBDSample(x=batch_x, y=batch_y, label=batch_label, mask=None)


@COLLATE_FUNCS.register_module()
class NTURGBDCollateFn:
    """
    Custom collate function for batched variable-length sequences.
    During the __call__ function, we creata `block_size`-long context windows, for each sequence in the batch.
    If is_packed is True, we pack the padded sequences, otherwise we return the padded sequences as is.

    Args:
    - block_size (int): Sequence's context length.
    - is_packed (bool): Whether to pack the padded sequence or not.

    Returns:
    
    The batched padded sequences ready to be fed to a transformer or an lstm model.
    """
    def __init__(self, block_size: int, is_packed: bool = False) -> None:
        self.block_size = block_size
        self.is_packed = is_packed
        
    def __call__(self, batch) -> NTURGBDSample:
        seq_lens = [sample.shape[0] for sample, _ in batch]
        labels = [label for _, label in batch]
        # A dataset's sample has a shape of (seq_len, n_bodies, n_joints, 3)
        # We want to create context windows of size `block_size` for each sample
        # and stack them together to form a batch of shape (batch_size, block_size, n_bodies, n_joints, 3)
        # We also want to create a target tensor of shape (batch_size, n_bodies, n_joints, 3)
        # The targets are shifted by 1 timestep to the right, so that the model can predict the next timestep
        batch_x = []
        batch_y = []
        for sample, _ in batch:
            x, y = self.get_windows(sample)
            chunk_size, context_len, n_bodies, n_joints, n_dims = x.shape
            batch_x.append(x.view(chunk_size, context_len, n_bodies * n_joints * n_dims))
            batch_y.append(y.view(chunk_size, context_len, n_bodies * n_joints * n_dims))
        # Pad the sequences to the maximum sequence length in the batch
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
        batch_y = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True)
        if self.is_packed:
            batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x, seq_lens, batch_first=True, enforce_sorted=False)
            batch_y = torch.nn.utils.rnn.pack_padded_sequence(batch_y, seq_lens, batch_first=True, enforce_sorted=False)
        labels = default_collate(labels)
        return NTURGBDSample(x=batch_x, y=batch_y, label=labels, mask=None)
    
    def get_windows(self, x):
        seq_len = x.shape[0]
        input_windows = []
        target_labels = []
        for i in range(seq_len - self.block_size):
            window = x[i:i + self.block_size, ...]
            target_label = x[i + 1:i + self.block_size + 1, ...]
            input_windows.append(window)
            target_labels.append(target_label)
        input_windows = np.array(input_windows)
        input_windows_tensor = torch.tensor(input_windows, dtype=torch.float)
        target_labels_tensor = torch.tensor(np.array(target_labels), dtype=torch.float)
        return input_windows_tensor, target_labels_tensor


@COLLATE_FUNCS.register_module()
class NTURGBDCollateFnWithRandomSampledContextWindow:
    """
    Custom collate function for batched variable-length sequences.
    During the __call__ function, we creata `block_size`-long context windows, for each sequence in the batch.
    If is_packed is True, we pack the padded sequences, otherwise we return the padded sequences as is.

    Args:
    - block_size (int): Sequence's context length.
    - is_packed (bool): Whether to pack the padded sequence or not.

    Returns:
    
    The batched padded sequences ready to be fed to a transformer or an lstm model.
    """
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        
    def __call__(self, batch) -> NTURGBDSample:
        # Pick a random index for each element of the batch and create a context window of size `block_size`
        # around that index
        # If the batch element's sequence length is less than `block_size`, then we sample the entire sequence
        # Pick the random index using pytorch
        seq_lens = [sample.shape[0] for sample, _ in batch]
        labels = [label for _, label in batch]
        pre_batch = []
        pre_mask = []
        for sample, _ in batch:
            if sample.shape[0] <= self.block_size:
                # Sample the entire sequence
                pre_batch.append(sample)
                pre_mask.append(torch.ones_like(sample))
            else:
                # Sample a random index
                idx = torch.randint(low=0, high=sample.shape[0] - self.block_size, size=(1,)).item()
                pre_batch.append(sample[idx:idx + self.block_size, ...])
                pre_mask.append(torch.ones_like(sample[idx:idx + self.block_size, ...]))
        # Pad the sequences to the maximum sequence length in the batch
        batch_x = torch.nn.utils.rnn.pad_sequence(pre_batch, batch_first=True)
        # Generate masks
        masks = torch.nn.utils.rnn.pad_sequence(pre_mask, batch_first=True)
        return NTURGBDSample(x=batch_x, y=batch_x, label=labels, mask=masks)
    

@DATASETS.register_module()
class NTURGBDDataset(Dataset):
    def __init__(
        self,
        data_directory: str,
        missing_files_dir: str = "../data/missing",
        label_file: str = '../data/labels.txt',
        max_context_window: int = 10,
        max_number_of_bodies: int = 4,
        max_duration: int = 300,
        n_joints: int = 25,
        transforms: Any = None,
        cache_file: str = None,
    ) -> None:
        self.data_directory = data_directory
        self.missing_files_dir = missing_files_dir
        self.labels_file = label_file
        self.labels_dict = dict()
        self.load_labels()
        self.max_context_window = max_context_window
        self.max_number_of_bodies = max_number_of_bodies
        self.max_duration = max_duration
        self.n_joints = n_joints
        self.transforms = transforms

        self.skeleton_files = get_skeleton_files(dataset_dir=self.data_directory)
        missing_files = get_missing_files(missing_files_dir=self.missing_files_dir)
        self.skeleton_files = filter_missing(
            missing_skeleton_names=missing_files, skeleton_files=self.skeleton_files
        )
        self.skeleton_files_clean = []

        if cache_file is None:
            for fname in self.skeleton_files:
                if should_blacklist(fname):
                    continue
                else:
                    self.skeleton_files_clean.append(fname)
        else:
            # Check if cache file exists and then unpickle it and store its data to self.skeleton_files_clean
            if os.path.exists(cache_file):
                # log that we are loading the cache file
                logging.info(f"Loading cache file {cache_file}...")
                with open(cache_file, 'rb') as f:
                    self.skeleton_files_clean = pickle.load(f)


    def load_labels(self):
        with open(self.labels_file, 'r') as f:
            for line in f:
                # Strip whitespace and split the line into the code and the label
                parts = line.strip().split('. ')
                if len(parts) == 2:
                    code, label = parts
                    # Map the code to a tuple of an integer (extracted from the code) and the label
                    self.labels_dict[code] = (int(code[1:])-1, label)


    def __getitem__(self, index) -> torch.Tensor:
        fname = self.skeleton_files_clean[index]
        # Get the label of the file from the filename
        activity_code_with_zeros = os.path.basename(fname).split('.')[0][-4:]
        activity_code = activity_code_with_zeros[0] + str(int(activity_code_with_zeros[1:]))
        label = self.labels_dict.get(activity_code, None)
        if label is None:
            raise ValueError(f"Label for activity code {activity_code} not found in label dictionary.")
        
        # Read the joints of the skeletons
        mat = read_skeleton_file(
            fname, save_skelxyz=True, save_rgbxy=False, save_depthxy=False
        )
        skels = []
        for i in range(self.max_number_of_bodies):
            if mat.get(f'skel_body{i}') is not None:
                skel = mat.get(f'skel_body{i}')
                skels.append(skel)
        
        skeletons_array = torch.from_numpy(np.array(skels)).permute(1, 0, 2, 3)
        if self.transforms:
            skeletons_array = self.transforms(skeletons_array)
        
        return skeletons_array, label


    def __len__(self):
        return len(self.skeleton_files_clean)
    
    def store_to_cache(self, cache_file: str) -> None:
        with open(cache_file, 'wb') as f:
            pickle.dump(self.skeleton_files_clean, f)
        logging.info(f"Stored {len(self.skeleton_files_clean)} files to cache file {cache_file}.")


@DATASETS.register_module()
class Human36mDataset(Dataset):
    def __init__(self, data_path, use_hourglass_detections=True, train=True) -> None:
        self.data_path = data_path
        self.use_hourglass_detections = use_hourglass_detections
        self.train = train

        self.train_inputs, self.test_inputs = [], []
        self.act = []

        if self.use_hourglass_detections:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
        else:
            train_2d_file = 'train_2d.pth.tar'
            train_2d_file = 'test_2d.pth.tar'
        
        if self.train:
            self.train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            self.train_2d = torch.load(os.path.join(data_path, train_2d_file))

            for k2d in self.train_2d.keys():
                (sub, act, fname) = k2d
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[0], f'(training) 3d and 2d shapes not matching'
                self.train_inputs.append(self.train_3d[k3d])
                self.act.append(act)

        else:
            self.test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d and 2d shapes not matching'
                self.test_inputs.append(self.test_3d[k3d])
                self.act.append(act)
    
    def __getitem__(self, index) -> Any:
        if self.train:
            # We want the sampeles to be returned as sequences
            # i.e.: [seq_len, n_joints, 3]
            x = torch.from_numpy(self.train_inputs[index]).float()
        else:
            x = torch.from_numpy(self.test_inputs[index]).float()
        return x.view(-1, 16, 3), self.act[index]

    def __len__(self):
        if self.train:
            return len(self.train_inputs)
        else:
            return len(self.test_inputs)
        

@DATASETS.register_module()
class AMASSTorchDataset(Dataset):
    def __init__(self,
                 index,
                 pose_data,
                 history_length: int,
                 prediction_horizon: int,
                 trans_data=None,
                 transform=lambda x: x,
                 window=None,
                 **kwargs):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform

        self._index = index
        self._pose_data = pose_data
        self._trans_data = trans_data
        self._window = window


        self.dataset_index = []

        self.index_data()

    def index_data(self):
        seq_lengths = self._index[:, 1] - self._index[:, 0]
        for j in range(seq_lengths.shape[0]):
            if self._window:
                for i in range(0, seq_lengths[j] - self._window[0] + 1, self._window[1]):
                    self.dataset_index += [(j, i)]
            else:
                for i in range(seq_lengths[j] - self.history_length - self.prediction_horizon):
                    self.dataset_index += [(j, i)]

    def __getitem__(self, item):
        i_idx, i = self.dataset_index[item]
        ts_pose = self._pose_data[self._index[i_idx, 0] + i: self._index[i_idx, 0] + i + self.history_length + self.prediction_horizon]
        ts_pose = self.transform(ts_pose)
        ts_pose = torch.tensor(ts_pose)

        if hasattr(self, 'dataset_name'):
            dp = (self.dataset_name, i_idx, i)
        else:
            dp = ("_", i_idx, i)

        if self._trans_data is not None:
            ts_trans = self._trans_data[self._index[i_idx, 0] + i: self._index[i_idx, 0] + i + self.history_length + self.prediction_horizon]
            ts_trans = torch.tensor(ts_trans)
            return (ts_pose[:self.history_length],
                    ts_pose[self.history_length: self.history_length + self.prediction_horizon],
                    ts_trans[:self.history_length],
                    ts_trans[self.history_length: self.history_length + self.prediction_horizon])

        return (ts_pose[:self.history_length],
                ts_pose[self.history_length: self.history_length + self.prediction_horizon])

    def __len__(self):
        return len(self.dataset_index)

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }