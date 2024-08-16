# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
import numpy as np
from typing import Dict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from mos4d.utils.cache import get_cache, memoize
from mos4d.config import MOS4DConfig, DataConfig, OdometryConfig
from mos4d.odometry import Odometry
from mos4d.datasets import dataset_factory, sequence_dataloaders


def collate_fn(batch):
    # Returns tensor of [batch, x, y, z, t, label]
    tensor_batch = None
    for i, past_point_clouds in enumerate(batch):
        ones = torch.ones(len(past_point_clouds), 1).type_as(past_point_clouds)
        tensor = torch.hstack([i * ones, past_point_clouds])
        tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])
    return tensor_batch


class MOS4DDataModule(LightningDataModule):
    """Training and validation set for Pytorch Lightning"""

    def __init__(self, dataloader: str, data_dir: Path, config: MOS4DConfig, cache_dir: Path):
        super(MOS4DDataModule, self).__init__()
        self.dataloader = dataloader
        self.data_dir = data_dir
        self.config = config
        self.cache_dir = cache_dir
        if self.cache_dir == None:
            print("No cache specified, therefore disabling shuffle during training!")
        self.shuffle = True if self.cache_dir is not None else False

        assert dataloader in sequence_dataloaders()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_set = MOS4DDataset(
            self.dataloader, self.data_dir, self.config, self.config.training.train, self.cache_dir
        )
        val_set = MOS4DDataset(
            self.dataloader, self.data_dir, self.config, self.config.training.val, self.cache_dir
        )
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.config.training.batch_size,
            collate_fn=collate_fn,
            shuffle=self.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

        self.valid_iter = iter(self.valid_loader)

        print(
            "Loaded {:d} training and {:d} validation samples.".format(len(train_set), len(val_set))
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class MOS4DDataset(Dataset):
    """Caches and returns scan and local maps for multiple sequences"""

    def __init__(
        self,
        dataloader: str,
        data_dir: Path,
        config: MOS4DConfig,
        sequences: list,
        cache_dir: Path,
    ):
        self.config = config
        self.sequences = sequences
        self._print = False

        # Cache
        if cache_dir is not None:
            directory = os.path.join(cache_dir, dataloader)
            self.use_cache = True
            self.cache = get_cache(directory=directory)
            print("Using cache at ", directory)
        else:
            self.use_cache = False
            self.cache = get_cache(directory=os.path.join(data_dir, "cache"))

        # Create datasets and map a sample index to the sequence and scan index
        self.datasets = {}
        self.idx_mapper = {}
        idx = 0
        for sequence in self.sequences:
            self.datasets[sequence] = dataset_factory(
                dataloader=dataloader,
                data_dir=data_dir,
                sequence=sequence,
            )
            for sample_idx in range(len(self.datasets[sequence])):
                self.idx_mapper[idx] = (sequence, sample_idx)
                idx += 1

        self.sequence = None
        self.odometry = Odometry(self.config.data, self.config.odometry)
        self.poses = []

    def __len__(self):
        return len(self.idx_mapper.keys())

    def __getitem__(self, idx):
        sequence, scan_index = self.idx_mapper[idx]
        return self.get_past_point_clouds(
            sequence,
            scan_index,
            self.config.mos.delay_mos,
            dict(self.config.data),
            dict(self.config.odometry),
        )

    @memoize()
    def get_past_point_clouds(
        self,
        sequence: int,
        scan_index: int,
        n_past_scans: int,
        data_config_dict: Dict,
        odometry_config_dict: Dict,
    ):
        """Returns up to n_past_scans in local frame and labels."""
        if not self._print:
            print("*****Caching now*****")
            self._print = True

        scan_points, timestamps, scan_labels = self.datasets[sequence][scan_index]

        # Only consider valid points
        valid_mask = scan_labels != -1
        scan_points = scan_points[valid_mask]
        scan_labels = scan_labels[valid_mask]

        if self.sequence != sequence or len(scan_points) == 0:
            data_config = DataConfig().model_validate(data_config_dict)
            odometry_config = OdometryConfig().model_validate(odometry_config_dict)

            self.sequence = sequence
            self.odometry = Odometry(data_config, odometry_config)
            self.poses = []

        # Register
        self.odometry.register_points(scan_points, timestamps, scan_index)
        self.poses.append(self.odometry.last_pose)

        # Collect past point clouds
        list_past_points = []
        n_poses_available = len(self.poses)
        for index in range(0, min(n_poses_available, n_past_scans)):
            past_points, _, past_labels = self.datasets[sequence][scan_index - index]

            valid_mask = past_labels != -1
            past_labels = past_labels[valid_mask]
            past_points = past_points[valid_mask]

            past_pose = self.poses[-(index + 1)]
            past_points_transformed = self.odometry.transform(
                past_points, np.linalg.inv(self.odometry.last_pose) @ past_pose
            )

            list_past_points.append(
                np.hstack(
                    [
                        past_points_transformed,
                        (scan_index - index) * np.ones((len(past_labels), 1)),
                        past_labels.reshape(-1, 1),
                    ]
                )
            )
        return torch.tensor(np.vstack(list_past_points)).to(torch.float32).reshape(-1, 5)
