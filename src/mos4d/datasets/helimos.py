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

import glob
import os

import numpy as np


class HeliMOSDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):
        self.sequence_id = sequence.split("/")[0]
        split_file = sequence.split("/")[1]
        self.sequence_dir = os.path.join(data_dir, self.sequence_id)
        self.scan_dir = os.path.join(self.sequence_dir, "velodyne/")

        self.scan_files = sorted(glob.glob(self.scan_dir + "*.bin"))
        self.calibration = self.read_calib_file(os.path.join(self.sequence_dir, "calib.txt"))

        # Load GT Poses (if available)
        self.poses_fn = os.path.join(self.sequence_dir, "poses.txt")
        if os.path.exists(self.poses_fn):
            self.gt_poses = self.load_poses(self.poses_fn)

        # No correction
        self.correct_kitti_scan = lambda frame: frame

        # Load labels
        self.label_dir = os.path.join(self.sequence_dir, "labels/")
        label_files = sorted(glob.glob(self.label_dir + "*.label"))

        # Get labels for train/val split if desired
        label_indices = np.loadtxt(os.path.join(data_dir, split_file), dtype=int).tolist()

        # Filter based on split if desired
        getIndex = lambda filename: int(os.path.basename(filename).split(".label")[0])
        self.dict_label_files = {
            getIndex(filename): filename
            for filename in label_files
            if getIndex(filename) in label_indices
        }

    def __getitem__(self, idx):
        points = self.scans(idx)
        timestamps = np.zeros(len(points))
        labels = (
            self.read_labels(self.dict_label_files[idx])
            if idx in self.dict_label_files.keys()
            else np.full(len(points), -1, dtype=np.int32)
        )
        return points, timestamps, labels

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame"""
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return Tr @ poses @ np.linalg.inv(Tr)

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        return points

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]

        # Ensure rotations are SO3
        rotations = poses[:, :3, :3]
        U, _, Vh = np.linalg.svd(rotations)
        poses[:, :3, :3] = U @ Vh

        return _lidar_pose_gt(poses)

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        orig_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
        orig_labels = orig_labels & 0xFFFF  # Mask semantics in lower half

        labels = np.zeros_like(orig_labels)
        labels[orig_labels <= 1] = -1  # Unlabeled (0), outlier (1)
        labels[orig_labels > 250] = 1  # Moving
        labels = labels.astype(dtype=np.int32).reshape(-1)
        return labels
