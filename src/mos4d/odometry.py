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
from mos4d.config import DataConfig, OdometryConfig

from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP


def parse_config(config_data: DataConfig, config_odometry: OdometryConfig):
    kiss_config = KISSConfig()
    kiss_config.data.deskew = config_data.deskew
    kiss_config.data.max_range = config_data.max_range
    kiss_config.data.min_range = config_data.min_range
    kiss_config.mapping.voxel_size = config_odometry.voxel_size
    kiss_config.mapping.max_points_per_voxel = config_odometry.max_points_per_voxel
    kiss_config.adaptive_threshold.initial_threshold = config_odometry.initial_threshold
    kiss_config.adaptive_threshold.min_motion_th = config_odometry.min_motion_th
    return kiss_config


class Odometry(KissICP):
    def __init__(
        self,
        config_data: DataConfig,
        config_odometry: OdometryConfig,
    ):
        kiss_config = parse_config(config_data, config_odometry)
        super().__init__(kiss_config)
        self.poses = []

    def register_points(self, points, timestamps, scan_index):
        # Apply motion compensation
        points = self.compensator.deskew_scan(points, timestamps, self.last_delta)

        # Preprocess the input cloud
        points_prep = self.preprocess(points)

        # Voxelize
        source, points_downsample = self.voxelize(points_prep)

        # Get motion prediction and adaptive_threshold
        sigma = self.adaptive_threshold.get_threshold()

        # Compute initial_guess for ICP
        initial_guess = self.last_pose @ self.last_delta

        new_pose = self.registration.align_points_to_map(
            points=source,
            voxel_map=self.local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma,
            kernel=sigma / 3,
        )

        # Compute the difference between the prediction and the actual estimate
        model_deviation = np.linalg.inv(initial_guess) @ new_pose

        # Update step: threshold, local map, delta, and the last pose
        self.adaptive_threshold.update_model_deviation(model_deviation)
        self.local_map.update(points_downsample, new_pose)
        self.last_delta = np.linalg.inv(self.last_pose) @ new_pose
        self.last_pose = new_pose

        points_reg = self.transform(points, self.last_pose)
        return np.asarray(points_reg)

    def transform(self, points, pose):
        points_hom = np.hstack((points, np.ones((len(points), 1))))
        points = (pose @ points_hom.T).T[:, :3]
        return points

    def current_pose(self):
        return self.last_pose

    def current_location(self):
        return self.last_pose[:3, 3]
