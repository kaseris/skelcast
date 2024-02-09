import copy

from dataclasses import dataclass

import numpy as np
import torch

from skelcast.data import DATASETS, COLLATE_FUNCS
from skelcast.data.human36m.camera import normalize_screen_coordinates
from skelcast.data.human36m.skeleton import Skeleton

from skelcast.data.human36m.quaternion import qeuler_np, qfix


class MocapDataset:
    def __init__(self, path, skeleton, fps):
        self._data = self._load(path)
        self._fps = fps
        self._use_gpu = False
        self._skeleton = skeleton

    def cuda(self):
        self._use_gpu = True
        self._skeleton.cuda()
        return self

    def _load(self, path):
        result = {}
        data = np.load(path, allow_pickle=True)
        for i, (trajectory, rotations, subject, action) in enumerate(
            zip(
                data["positions"],
                data["rotations"],
                data["subjects"],
                data["actions"],
            )
        ):
            if subject not in result:
                result[subject] = {}

            result[subject][action] = {"rotations": rotations, "trajectory": trajectory}
        return result

    def downsample(self, factor, keep_strides=True):
        """
        Downsample this dataset by an integer factor, keeping all strides of the data
        if keep_strides is True.
        The frame rate must be divisible by the given factor.
        The sequences will be replaced by their downsampled versions, whose actions
        will have '_d0', ... '_dn' appended to their names.
        """
        assert self._fps % factor == 0

        for subject in self._data.keys():
            new_actions = {}
            for action in list(self._data[subject].keys()):
                for idx in range(factor):
                    tup = {}
                    for k in self._data[subject][action].keys():
                        tup[k] = self._data[subject][action][k][idx::factor]
                    new_actions[action + "_d" + str(idx)] = tup
                    if not keep_strides:
                        break
            self._data[subject] = new_actions

        self._fps //= factor

    def _mirror_sequence(self, sequence):
        mirrored_rotations = sequence["rotations"].copy()
        mirrored_trajectory = sequence["trajectory"].copy()

        joints_left = self._skeleton.joints_left()
        joints_right = self._skeleton.joints_right()

        # Flip left/right joints
        mirrored_rotations[:, joints_left] = sequence["rotations"][:, joints_right]
        mirrored_rotations[:, joints_right] = sequence["rotations"][:, joints_left]

        mirrored_rotations[:, :, [2, 3]] *= -1
        mirrored_trajectory[:, 0] *= -1

        return {
            "rotations": qfix(mirrored_rotations),
            "trajectory": mirrored_trajectory,
        }

    def mirror(self):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        The mirrored sequences will have '_m' appended to the action name.
        """
        for subject in self._data.keys():
            for action in list(self._data[subject].keys()):
                if "_m" in action:
                    continue
                self._data[subject][action + "_m"] = self._mirror_sequence(
                    self._data[subject][action]
                )

    def compute_euler_angles(self, order):
        for subject in self._data.values():
            for action in subject.values():
                action["rotations_euler"] = qeuler_np(
                    action["rotations"], order, use_gpu=self._use_gpu
                )

    def compute_positions(self):
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(
                    action["rotations"].astype("float32")
                ).unsqueeze(0)
                trajectory = torch.from_numpy(
                    action["trajectory"].astype("float32")
                ).unsqueeze(0)
                if self._use_gpu:
                    rotations = rotations.cuda()
                    trajectory = trajectory.cuda()
                action["positions_world"] = (
                    self._skeleton.forward_kinematics(rotations, trajectory)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                # Absolute translations across the XY plane are removed here
                trajectory[:, :, [0, 2]] = 0
                action["positions_local"] = (
                    self._skeleton.forward_kinematics(rotations, trajectory)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

    # def __getitem__(self, key):
    #     return self._data[key]

    def subjects(self):
        return self._data.keys()

    def subject_actions(self, subject):
        return self._data[subject].keys()

    def all_actions(self):
        result = []
        for subject, actions in self._data.items():
            for action in actions.keys():
                result.append((subject, action))
        return result

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton


@dataclass
class Human36MSample:
    x: torch.tensor
    y: torch.tensor
    mask: torch.tensor = None


@COLLATE_FUNCS.register_module()
class Human36MCollateFnWithRandomSampledContextWindow:
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

    def __call__(self, batch) -> Human36MSample:
        # Pick a random index for each element of the batch and create a context window of size `block_size`
        # around that index
        # If the batch element's sequence length is less than `block_size`, then we sample the entire sequence
        # Pick the random index using pytorch
        seq_lens = [sample.shape[0] for sample in batch]
        pre_batch = []
        pre_mask = []
        for sample in batch:
            if sample.shape[0] <= self.block_size:
                # Sample the entire sequence
                pre_batch.append(sample)
                pre_mask.append(torch.ones_like(sample))
            else:
                # Sample a random index
                idx = torch.randint(
                    low=0, high=sample.shape[0] - self.block_size, size=(1,)
                ).item()
                pre_batch.append(torch.from_numpy(sample[idx : idx + self.block_size, ...]))
        # Pad the sequences to the maximum sequence length in the batch
        batch_x = torch.nn.utils.rnn.pad_sequence(pre_batch, batch_first=True)
        # Generate masks
        return Human36MSample(x=batch_x, y=batch_x, mask=None)


@DATASETS.register_module()
class Human36MDataset(MocapDataset):
    """
    TODO: Possibly add a flatten_data method for easy accessing with a single index.
    """

    def __init__(self, path, seq_len=27, **kwargs):
        skeleton = Skeleton(
            offsets=[
                [0.0, 0.0, 0.0],
                [-132.948591, 0.0, 0.0],
                [0.0, -442.894612, 0.0],
                [0.0, -454.206447, 0.0],
                [0.0, 0.0, 162.767078],
                [0.0, 0.0, 74.999437],
                [132.948826, 0.0, 0.0],
                [0.0, -442.894413, 0.0],
                [0.0, -454.20659, 0.0],
                [0.0, 0.0, 162.767426],
                [0.0, 0.0, 74.999948],
                [0.0, 0.1, 0.0],
                [0.0, 233.383263, 0.0],
                [0.0, 257.077681, 0.0],
                [0.0, 121.134938, 0.0],
                [0.0, 115.002227, 0.0],
                [0.0, 257.077681, 0.0],
                [0.0, 151.034226, 0.0],
                [0.0, 278.882773, 0.0],
                [0.0, 251.733451, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 99.999627],
                [0.0, 100.000188, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 257.077681, 0.0],
                [0.0, 151.031437, 0.0],
                [0.0, 278.892924, 0.0],
                [0.0, 251.72868, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 99.999888],
                [0.0, 137.499922, 0.0],
                [0.0, 0.0, 0.0],
            ],
            parents=[
                -1,
                0,
                1,
                2,
                3,
                4,
                0,
                6,
                7,
                8,
                9,
                0,
                11,
                12,
                13,
                14,
                12,
                16,
                17,
                18,
                19,
                20,
                19,
                22,
                12,
                24,
                25,
                26,
                27,
                28,
                27,
                30,
            ],
            joints_left=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
            joints_right=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
        )
        super().__init__(path, skeleton, fps=50)
        self.compute_positions()
        self._dataset_flat = []
        for subject in ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]:
            for action in list(self._data[subject].keys()):
                self._dataset_flat.append(self._data[subject][action]["rotations"])

    def __getitem__(self, index):
        return self._dataset_flat[index]

    def __len__(self):
        return len(self._dataset_flat)
