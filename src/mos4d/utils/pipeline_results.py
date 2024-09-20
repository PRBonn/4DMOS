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
from kiss_icp.metrics import absolute_trajectory_error, sequence_error
from kiss_icp.tools.pipeline_results import PipelineResults

from mos4d.metrics import get_f1, get_iou, get_precision, get_recall, get_stats


class MOSPipelineResults(PipelineResults):
    def __init__(self) -> None:
        super().__init__()

    def eval_odometry(self, gt_poses, poses):
        avg_tra, avg_rot = sequence_error(gt_poses, poses)
        ate_rot, ate_trans = absolute_trajectory_error(gt_poses, poses)
        self.append(desc="Average Translation Error", units="%", value=avg_tra)
        self.append(desc="Average Rotational Error", units="deg/m", value=avg_rot)
        self.append(desc="Absoulte Trajectory Error (ATE)", units="m", value=ate_trans)
        self.append(desc="Absoulte Rotational Error (ARE)\n", units="rad", value=ate_rot)

    def eval_mos(self, confusion_matrix, desc=""):
        iou = get_iou(confusion_matrix)
        tp, fp, fn = get_stats(confusion_matrix)
        recall = get_recall(confusion_matrix)
        precision = get_precision(confusion_matrix)
        f1 = get_f1(confusion_matrix)
        self.append(desc=desc, units="", value="")
        self.append(desc="Moving IoU", units="%", value=iou[1].item() * 100)
        self.append(desc="Moving Recall", units="%", value=recall[1].item() * 100)
        self.append(desc="Moving Precision", units="%", value=precision[1].item() * 100)
        self.append(desc="Moving F1", units="%", value=f1[1].item() * 100)
        self.append(desc="Moving TP", units="points", value=int(tp[1].item()))
        self.append(desc="Moving FP", units="points", value=int(fp[1].item()))
        self.append(desc="Moving FN\n", units="points", value=int(fn[1].item()))

    def eval_fps(self, times, desc="Average Frequency"):
        def _get_fps(times):
            total_time_s = sum(times) * 1e-9
            return float(len(times) / total_time_s)

        avg_fps_mos = int(np.floor(_get_fps(times)))
        self.append(
            desc=f"{desc}",
            units="Hz",
            value=avg_fps_mos,
            trunc=True,
        )
