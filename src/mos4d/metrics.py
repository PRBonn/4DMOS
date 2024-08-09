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

import torch


def get_confusion_matrix(pred_labels, gt_labels):
    # Mask valid values -1
    valid_mask = gt_labels != -1
    pred_labels = pred_labels[valid_mask]
    gt_labels = gt_labels[valid_mask]

    idxs = torch.stack([pred_labels, gt_labels], dim=0).long()
    ones = torch.ones((idxs.shape[-1])).type_as(gt_labels)
    confusion_matrix = torch.zeros(2, 2).type_as(gt_labels)
    confusion_matrix = confusion_matrix.index_put_(tuple(idxs), ones, accumulate=True)
    return confusion_matrix


def get_stats(confusion_matrix):
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=1) - tp
    fn = confusion_matrix.sum(dim=0) - tp
    return tp, fp, fn


def get_iou(confusion_matrix):
    tp, fp, fn = get_stats(confusion_matrix)
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union

    # If no GT labels, set to NaN
    iou[tp + fn == 0] = float("nan")

    return iou


def get_precision(confusion_matrix):
    tp, fp, fn = get_stats(confusion_matrix)
    prec = tp / (tp + fp)

    # If no GT labels, set to NaN
    prec[tp + fn == 0] = float("nan")

    return prec


def get_recall(confusion_matrix):
    tp, _, fn = get_stats(confusion_matrix)
    rec = tp / (tp + fn)

    # If no GT labels, set to NaN
    rec[tp + fn == 0] = float("nan")

    return rec


def get_f1(confusion_matrix):
    if torch.sum(confusion_matrix) == 0:
        return torch.tensor([float("nan"), float("nan")])
    prec = get_precision(confusion_matrix)
    rec = get_recall(confusion_matrix)
    f1 = 2 * prec * rec / (prec + rec)

    # If no GT labels or zero Prec and Recall, set to NaN
    f1[(prec == -1) | (rec == -1)] = float("nan")
    f1[prec + rec == 0] = float("nan")

    return f1


def get_acc(confusion_matrix):
    if torch.sum(confusion_matrix) == 0:
        return torch.tensor([-1, -1])
    tp, fp, _ = get_stats(confusion_matrix)
    total_tp = tp.sum()
    total = tp.sum() + fp.sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean
