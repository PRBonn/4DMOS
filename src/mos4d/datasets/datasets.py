#!/usr/bin/env python3
# @file      datasets.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import numpy as np
import yaml
import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from mos4d.datasets.utils import load_poses, load_calib, load_files
from mos4d.datasets.augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)


class KittiSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential KITTI data"""

    def __init__(self, cfg):
        """Method to initizalize the KITTI dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        super(KittiSequentialModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""

        ########## Point dataset splits
        train_set = KittiSequentialDataset(self.cfg, split="train")

        val_set = KittiSequentialDataset(self.cfg, split="val")

        test_set = KittiSequentialDataset(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):
        meta = [item[0] for item in batch]
        past_point_clouds = [item[1] for item in batch]
        past_labels = [item[2] for item in batch]
        return [meta, past_point_clouds, past_labels]


class KittiSequentialDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("DATA")

        # Pose information
        self.transform = self.cfg["DATA"]["TRANSFORM"]
        self.poses = {}
        self.filename_poses = cfg["DATA"]["POSES"]

        # Semantic information
        self.semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]

        self.split = split
        if self.split == "train":
            self.sequences = self.cfg["DATA"]["SPLIT"]["TRAIN"]
        elif self.split == "val":
            self.sequences = self.cfg["DATA"]["SPLIT"]["VAL"]
        elif self.split == "test":
            self.sequences = self.cfg["DATA"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.dt_data = self.cfg["DATA"]["DELTA_T_DATA"]
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"

        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames = {}

        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0
        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            path_to_seq = os.path.join(self.root_dir, seqstr)

            scan_path = os.path.join(path_to_seq, "velodyne")
            self.filenames[seq] = load_files(scan_path)
            if self.transform:
                self.poses[seq] = self.read_poses(path_to_seq)
                assert len(self.poses[seq]) == len(self.filenames[seq])
            else:
                self.poses[seq] = []

            # Get number of sequences based on number of past steps
            n_samples_sequence = max(
                0, len(self.filenames[seq]) - self.skip * (self.n_past_steps - 1)
            )

            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.skip * (self.n_past_steps - 1) + sample_idx
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Load point clouds and get sequence

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        seq, scan_idx = self.idx_mapper[idx]

        # Load past point clouds
        from_idx = scan_idx - self.skip * (self.n_past_steps - 1)
        to_idx = scan_idx + 1
        past_indices = list(range(from_idx, to_idx, self.skip))
        past_files = self.filenames[seq][from_idx : to_idx : self.skip]
        list_past_point_clouds = [self.read_point_cloud(f) for f in past_files]
        for i, pcd in enumerate(list_past_point_clouds):
            # Transform to current viewpoint
            if self.transform:
                from_pose = self.poses[seq][past_indices[i]]
                to_pose = self.poses[seq][past_indices[-1]]
                pcd = self.transform_point_cloud(pcd, from_pose, to_pose)
            time_index = i - self.n_past_steps + 1
            timestamp = round(time_index * self.dt_pred, 3)
            list_past_point_clouds[i] = self.timestamp_tensor(pcd, timestamp)

        past_point_clouds = torch.cat(list_past_point_clouds, dim=0)

        # Load past labels
        label_files = [
            os.path.join(self.root_dir, str(seq).zfill(2), "labels", str(i).zfill(6) + ".label")
            for i in past_indices
        ]

        list_past_labels = [self.read_labels(f) for f in label_files]
        for i, labels in enumerate(list_past_labels):
            time_index = i - self.n_past_steps + 1
            timestamp = round(time_index * self.dt_pred, 3)
            list_past_labels[i] = self.timestamp_tensor(labels, timestamp)
        past_labels = torch.cat(list_past_labels, dim=0)

        if self.augment:
            past_point_clouds, past_labels = self.augment_data(past_point_clouds, past_labels)

        meta = (seq, scan_idx, past_indices)
        return [meta, past_point_clouds, past_labels]

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
        NP = past_point_clouds.shape[0]
        xyz1 = torch.hstack([past_point_clouds, torch.ones(NP, 1)]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def augment_data(self, past_point_clouds, past_labels):
        past_point_clouds = rotate_point_cloud(past_point_clouds)
        past_point_clouds = rotate_perturbation_point_cloud(past_point_clouds)
        past_point_clouds = jitter_point_cloud(past_point_clouds)
        past_point_clouds = shift_point_cloud(past_point_clouds)
        past_point_clouds = random_flip_point_cloud(past_point_clouds)
        past_point_clouds = random_scale_point_cloud(past_point_clouds)
        return past_point_clouds, past_labels

    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
        point_cloud = point_cloud[:, :3]
        return point_cloud

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        if os.path.isfile(filename):
            labels = np.fromfile(filename, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for k, v in self.semantic_config["learning_map"].items():
                mapped_labels[labels == k] = v
            selected_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
            selected_labels = selected_labels.reshape((-1, 1))
            return selected_labels
        else:
            return torch.Tensor(1, 1).long()

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
        calib_file = os.path.join(path_to_seq, "calib.txt")
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        poses = np.array(new_poses)
        return poses
