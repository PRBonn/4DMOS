#!/usr/bin/env python3
# @file      metrics.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationMetrics(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def compute_confusion_matrix(
        self, pred_logits: torch.Tensor, gt_labels: torch.Tensor
    ):

        # Set ignored classes to -inf to not influence softmax
        pred_logits[:, self.ignore_index] = -float("inf")

        pred_softmax = F.softmax(pred_logits, dim=1)
        pred_labels = torch.argmax(pred_softmax, axis=1).long()
        gt_labels = gt_labels.long()

        idxs = torch.stack([pred_labels, gt_labels], dim=0)
        ones = torch.ones((idxs.shape[-1])).type_as(gt_labels)
        confusion_matrix = torch.zeros(self.n_classes, self.n_classes).type_as(
            gt_labels
        )
        confusion_matrix = confusion_matrix.index_put_(
            tuple(idxs), ones, accumulate=True
        )
        return confusion_matrix

    def getStats(self, confusion_matrix):
        ignore_mask = torch.Tensor(self.ignore_index).long()
        confusion_matrix[:, ignore_mask] = 0

        tp = confusion_matrix.diag()
        fp = confusion_matrix.sum(dim=1) - tp
        fn = confusion_matrix.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        return iou

    def getacc(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        total_tp = tp.sum()
        total = tp.sum() + fp.sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean
