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
from mos4d.datasets.kitti import SemanticKITTIDataset


class SemanticApolloDataset(SemanticKITTIDataset):
    def __init__(self, data_dir, sequence: str, *_, **__):
        super().__init__(data_dir=data_dir, sequence=sequence, *_, **__)
        self.correct_kitti_scan = lambda frame: frame

    def get_frames_timestamps(self) -> np.ndarray:
        return np.arange(0, self.__len__(), 1.0)

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        orig_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
        orig_labels = orig_labels & 0xFFFF  # Mask semantics in lower half
        labels = np.zeros_like(orig_labels)
        labels[orig_labels > 250] = 1  # Moving
        labels = labels.astype(dtype=np.int32).reshape(-1)
        return labels
