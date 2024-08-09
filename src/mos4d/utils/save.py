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

import numpy as np
import os
import importlib
from pathlib import Path


class StubWriter:
    def __init__(self) -> None:
        pass

    def write(self, *_, **__):
        pass


class PlyWriter:
    def __init__(self) -> None:
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

    def write(
        self, scan_points: np.ndarray, pred_labels: np.ndarray, gt_labels: np.ndarray, filename: str
    ):
        os.makedirs(Path(filename).parent, exist_ok=True)
        pcd_current_scan = self.o3d.geometry.PointCloud(
            self.o3d.utility.Vector3dVector(scan_points)
        ).paint_uniform_color([0, 0, 0])

        scan_colors = np.array(pcd_current_scan.colors)

        tp = (pred_labels == 1) & (gt_labels == 1)
        fp = (pred_labels == 1) & (gt_labels != 1)
        fn = (pred_labels != 1) & (gt_labels == 1)

        scan_colors[tp] = [0, 1, 0]
        scan_colors[fp] = [1, 0, 0]
        scan_colors[fn] = [0, 0, 1]

        pcd_current_scan.colors = self.o3d.utility.Vector3dVector(scan_colors)
        self.o3d.io.write_point_cloud(filename, pcd_current_scan)


class KITTIWriter:
    def __init__(self) -> None:
        pass

    def write(self, pred_labels: np.ndarray, filename: str):
        os.makedirs(Path(filename).parent, exist_ok=True)
        kitti_labels = np.copy(pred_labels)
        kitti_labels[pred_labels == 0] = 9
        kitti_labels[pred_labels == 1] = 251
        kitti_labels = kitti_labels.reshape(-1).astype(np.int32)
        kitti_labels.tofile(filename)
