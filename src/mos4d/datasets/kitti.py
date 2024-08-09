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
from kiss_icp.datasets.kitti import KITTIOdometryDataset


class SemanticKITTIDataset(KITTIOdometryDataset):
    def __init__(self, data_dir, sequence: str, *_, **__):
        self.sequence_id = sequence.zfill(2)
        self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

        # Load GT Poses (if available)
        self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
        if os.path.exists(self.poses_fn):
            self.gt_poses = self.load_poses(self.poses_fn)

        # Add correction for KITTI datasets, can be easilty removed if unwanted
        from kiss_icp.pybind import kiss_icp_pybind

        self.correct_kitti_scan = lambda frame: np.asarray(
            kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
        )

        self.label_dir = os.path.join(self.kitti_sequence_dir, "labels/")
        self.label_files = sorted(glob.glob(self.label_dir + "*.label"))

    def __getitem__(self, idx):
        points = self.scans(idx)
        timestamps = np.zeros(len(points))
        labels = (
            self.read_labels(self.label_files[idx])
            if self.label_files
            else np.full((len(points), 1), -1, dtype=np.int32)
        )
        return points, timestamps, labels

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        orig_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
        orig_labels = orig_labels & 0xFFFF  # Mask semantics in lower half
        labels = np.zeros_like(orig_labels)
        labels[orig_labels <= 1] = -1  # Unlabeled (0), outlier (1)
        labels[orig_labels > 250] = 1  # Moving
        labels = labels.astype(dtype=np.int32).reshape(-1)
        return labels
