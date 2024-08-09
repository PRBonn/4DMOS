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
import time
from pathlib import Path
from typing import Optional
from collections import deque
import torch
import numpy as np
from tqdm.auto import trange

from kiss_icp.pipeline import OdometryPipeline

from mos4d.mos4d import MOS4DNet
from mos4d.odometry import Odometry
from mos4d.metrics import get_confusion_matrix
from mos4d.utils.visualizer import MOS4DVisualizer, StubVisualizer
from mos4d.utils.pipeline_results import MOSPipelineResults
from mos4d.utils.save import KITTIWriter, StubWriter
from mos4d.config import load_config


def prob_to_log_odds(prob):
    odds = np.divide(prob, 1 - prob + 1e-10)
    log_odds = np.log(odds)
    return log_odds


class MOS4DPipeline(OdometryPipeline):
    def __init__(
        self,
        dataset,
        weights: Path,
        config: Optional[Path] = None,
        visualize: bool = False,
        save_kitti: bool = False,
        n_scans: int = -1,
        jump: int = 0,
    ):
        self._dataset = dataset
        self._n_scans = (
            len(self._dataset) - jump if n_scans == -1 else min(len(self._dataset) - jump, n_scans)
        )
        self._first = jump
        self._last = self._first + self._n_scans

        # Config and output dir
        self.config = load_config(config)
        self.results_dir = None

        # Pipeline
        state_dict = {
            k.replace("model.", ""): v for k, v in torch.load(weights)["state_dict"].items()
        }
        state_dict = {k.replace("mos.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}

        # Change these depending on the 4DMOS model
        self.prior = 0.25
        self.max_length = 10
        self.mos_voxel_size = 0.1

        self.model = MOS4DNet(self.mos_voxel_size)
        self.model.load_state_dict(state_dict)
        self.model.cuda().eval().freeze()

        self.odometry = Odometry(self.config.data, self.config.odometry)
        self.buffer = deque(maxlen=self.max_length)
        self.dict_logits = {}
        self.dict_gt_labels = {}

        # Results
        self.results = MOSPipelineResults()
        self.poses = self.odometry.poses
        self.has_gt = hasattr(self._dataset, "gt_poses")
        self.gt_poses = self._dataset.gt_poses[self._first : self._last] if self.has_gt else None
        self.dataset_name = self._dataset.__class__.__name__
        self.dataset_sequence = (
            self._dataset.sequence_id
            if hasattr(self._dataset, "sequence_id")
            else os.path.basename(self._dataset.data_dir)
        )
        self.config.out_dir += f"/4dmos/{self.dataset_sequence}"

        self.confusion_matrix_online = torch.zeros(2, 2)
        self.confusion_matrix_receding = torch.zeros(2, 2)
        self.times_mos = []

        # Visualizer
        self.visualize = visualize
        self.visualizer = MOS4DVisualizer() if visualize else StubVisualizer()

        self.kitti_writer = KITTIWriter() if save_kitti else StubWriter()

    # Public interface  ------
    def run(self):
        self._create_output_dir()
        with torch.no_grad():
            self._run_pipeline()
        self._run_evaluation()
        self._write_result_poses()
        self._write_gt_poses()
        self._write_cfg()
        self._write_log()
        return self.results

    def _preprocess(self, points, min_range, max_range):
        ranges = np.linalg.norm(points - self.odometry.current_location(), axis=1)
        mask = ranges <= max_range if max_range > 0 else np.ones_like(ranges, dtype=bool)
        mask = np.logical_and(mask, ranges >= min_range)
        return mask

    # Private interface  ------
    def _run_pipeline(self):
        pbar = trange(self._first, self._last, unit=" frames", dynamic_ncols=True)
        for scan_index in pbar:
            local_scan, timestamps, gt_labels = self._next(scan_index)
            scan_points = self.odometry.register_points(local_scan, timestamps, scan_index)

            self.dict_gt_labels[scan_index] = gt_labels

            scan_points = torch.tensor(scan_points, dtype=torch.float32, device="cuda")
            self.buffer.append(
                torch.hstack(
                    [
                        scan_points,
                        scan_index
                        * torch.ones(len(scan_points)).reshape(-1, 1).type_as(scan_points),
                    ]
                )
            )

            past_point_clouds = torch.vstack(list(self.buffer))
            start_time = time.perf_counter_ns()
            pred_logits = self.model.predict(past_point_clouds)
            self.times_mos.append(time.perf_counter_ns() - start_time)

            # Detach, move to CPU
            pred_logits = pred_logits.detach().cpu().numpy().astype(np.float64)
            scan_points = scan_points.cpu().numpy().astype(np.float64)
            past_point_clouds = past_point_clouds.cpu().numpy().astype(np.float64)
            torch.cuda.empty_cache()

            # Fuse predictions in binary Bayes filter
            for past_scan_index in np.unique(past_point_clouds[:, -1]):
                mask_past_scan = past_point_clouds[:, -1] == past_scan_index
                scan_logits = pred_logits[mask_past_scan]

                if past_scan_index not in self.dict_logits.keys():
                    self.dict_logits[past_scan_index] = scan_logits
                else:
                    self.dict_logits[past_scan_index] += scan_logits
                    self.dict_logits[past_scan_index] -= prob_to_log_odds(self.prior)

            pred_labels = self.model.to_label(pred_logits)

            mask_scan = past_point_clouds[:, -1] == scan_index
            past_point_clouds_global = self.odometry.transform(
                past_point_clouds[:, 1:4], self.odometry.current_pose()
            )
            if self.visualize:
                self.visualizer.update(
                    scan_points,
                    past_point_clouds_global[~mask_scan],
                    pred_labels[mask_scan],
                    pred_labels[~mask_scan],
                    pred_labels[mask_scan],
                    pred_labels[~mask_scan],
                    self.odometry.current_pose(),
                )

            self.confusion_matrix_online += get_confusion_matrix(
                torch.tensor(pred_labels[mask_scan], dtype=torch.int32),
                torch.tensor(gt_labels, dtype=torch.int32),
            )
            self.kitti_writer.write(
                pred_labels[mask_scan],
                filename=f"{self.results_dir}/4dmos_online/bin/sequences/{self.dataset_sequence}/predictions/{scan_index:06}.label",
            )

        # Get final predictions from filter
        pbar = trange(self._first, self._last, desc="Fusing", unit=" frames", dynamic_ncols=True)
        for scan_index in pbar:
            pred_labels = self.model.to_label(self.dict_logits[scan_index])
            gt_labels = self.dict_gt_labels[scan_index].reshape(-1)
            self.confusion_matrix_receding += get_confusion_matrix(
                torch.tensor(pred_labels, dtype=torch.int32),
                torch.tensor(gt_labels, dtype=torch.int32),
            )
            self.kitti_writer.write(
                pred_labels,
                filename=f"{self.results_dir}/4dmos_receding/bin/sequences/{self.dataset_sequence}/predictions/{scan_index:06}.label",
            )

    def _next(self, idx):
        dataframe = self._dataset[idx]
        try:
            local_scan, timestamps, gt_labels = dataframe
        except ValueError:
            try:
                local_scan, timestamps = dataframe
                gt_labels = -1 * np.ones(local_scan.shape[0])
            except ValueError:
                local_scan = dataframe
                gt_labels = -1 * np.ones(local_scan.shape[0])
                timestamps = np.zeros(local_scan.shape[0])
        return local_scan.reshape(-1, 3), timestamps.reshape(-1), gt_labels.reshape(-1)

    def _run_evaluation(self):
        if self.has_gt:
            self.results.eval_odometry(self.odometry.get_poses(), self.gt_poses)
        self.results.eval_mos(self.confusion_matrix_online, desc="Online Prediction")
        self.results.eval_mos(self.confusion_matrix_receding, desc="Receding Horizon Strategy")
        self.results.eval_fps(self.times_mos, desc="Average Frequency 4DMOS")
