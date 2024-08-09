#!/usr/bin/env python3
# @file      mos4d.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved
import torch
import torch.nn as nn
import copy
import numpy as np
import MinkowskiEngine as ME
from pytorch_lightning import LightningModule

from mos4d.minkunet import CustomMinkUNet14


class MOS4DNet(LightningModule):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size
        self.MinkUNet = CustomMinkUNet14(in_channels=1, out_channels=3, D=4)
        self.softmax = nn.Softmax(dim=1)

    def predict(self, past_point_clouds: torch.Tensor):
        coordinates = torch.hstack(
            [torch.zeros(len(past_point_clouds), 1).type_as(past_point_clouds), past_point_clouds]
        )
        logits = self.forward(coordinates)
        return self.to_single_logit(logits)

    def forward(self, coordinates: torch.Tensor):
        quantization = torch.Tensor(
            [1.0, self.voxel_size, self.voxel_size, self.voxel_size, 1.0]
        ).type_as(coordinates)
        coordinates = torch.div(coordinates, quantization)
        features = 0.5 * torch.ones(len(coordinates), 1).type_as(coordinates)

        tensor_field = ME.TensorField(features=features, coordinates=coordinates.type_as(features))
        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)
        out = predicted_sparse_tensor.slice(tensor_field)
        out.features[:, 0] = -float("inf")
        return out.features

    def to_single_logit(self, logits: torch.Tensor):
        softmax = self.softmax(logits)
        return self.prob_to_log_odds(softmax)[:, 2]

    @staticmethod
    def prob_to_log_odds(prob):
        odds = torch.divide(prob, 1 - prob + 1e-10)
        log_odds = torch.log(odds)
        return log_odds

    @staticmethod
    def to_label(logits):
        labels = copy.deepcopy(logits)
        mask = logits > 0
        labels[mask] = 1.0
        labels[~mask] = 0.0
        return labels
