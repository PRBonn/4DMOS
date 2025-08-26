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
import torch
from pytorch_lightning import LightningModule

from mos4d.config import MOS4DConfig
from mos4d.metrics import get_confusion_matrix, get_iou, get_precision, get_recall
from mos4d.mos4d import MOS4DNet
from mos4d.utils.augmentation import (
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
    rotate_point_cloud,
)


class TrainingModule(LightningModule):
    def __init__(self, config: MOS4DConfig):
        super().__init__()
        self.save_hyperparameters(dict(config))
        self.batch_size = config.training.batch_size
        self.lr = config.training.lr
        self.lr_epoch = config.training.lr_epoch
        self.lr_decay = config.training.lr_decay
        self.weight_decay = config.training.weight_decay
        self.mos = MOS4DNet(config.mos.voxel_size_mos)
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_reset()
        self.val_reset()

    def train_reset(self):
        self.train_confusion_matrix = torch.zeros(2, 2)

    def val_reset(self):
        self.val_confusion_matrix = torch.zeros(2, 2)

    def training_step(self, batch: torch.Tensor, batch_idx, dataloader_index=0):
        # Batch is [batch,x,y,z,t,label]

        # Skip step if too few moving points
        num_moving_points = len(batch[batch[:, -1] == 1.0])
        num_points = len(batch)
        if num_points == 0 or num_moving_points / num_points < 0.001:
            return None

        batch = self.augmentation(batch)

        # Only train if enough points are left
        if len(batch) < 100:
            return None

        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, -1].reshape(-1)
        logits = self.mos.forward(coordinates)

        gt_indices = (gt_labels + 1).to(torch.long)
        loss = self.loss(logits, gt_indices)
        self.log("train_loss", loss.item(), on_step=True)

        logits = logits.detach().cpu()
        gt_labels = gt_labels.detach().cpu()

        pred_logits = self.mos.to_single_logit(logits)
        pred_labels = self.mos.to_label(pred_logits)

        self.train_confusion_matrix += get_confusion_matrix(pred_labels, gt_labels)

        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        iou = get_iou(self.train_confusion_matrix)
        recall = get_recall(self.train_confusion_matrix)
        precision = get_precision(self.train_confusion_matrix)
        self.log("train_moving_iou", iou[1].item())
        self.log("train_moving_recall", recall[1].item())
        self.log("train_moving_precision", precision[1].item())
        self.train_reset()
        torch.cuda.empty_cache()

    def validation_step(self, batch: torch.Tensor, batch_idx):
        # Batch is [batch,x,y,z,t,label]
        if len(batch) < 100:
            return None

        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, -1].reshape(-1)

        logits = self.mos.forward(coordinates)

        gt_indices = (gt_labels + 1).to(torch.long)
        loss = self.loss(logits, gt_indices)

        self.log("val_loss", loss.item(), batch_size=len(batch), prog_bar=True, on_epoch=True)

        logits = logits.detach().cpu()
        gt_labels = gt_labels.detach().cpu()

        pred_logits = self.mos.to_single_logit(logits)
        pred_labels = self.mos.to_label(pred_logits)

        # Logging metrics
        self.val_confusion_matrix += get_confusion_matrix(pred_labels, gt_labels)
        torch.cuda.empty_cache()
        return loss

    def on_validation_epoch_end(self):
        iou = get_iou(self.val_confusion_matrix)
        recall = get_recall(self.val_confusion_matrix)
        precision = get_precision(self.val_confusion_matrix)
        self.log("val_moving_iou", iou[1].item())
        self.log("val_moving_recall", recall[1].item())
        self.log("val_moving_precision", precision[1].item())
        self.val_reset()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_epoch, gamma=self.lr_decay
        )
        return [optimizer], [scheduler]

    def augmentation(self, batch):
        batch = self.crop(batch)
        batch[:, 1:4] = rotate_point_cloud(batch[:, 1:4])
        batch[:, 1:4] = rotate_perturbation_point_cloud(batch[:, 1:4])
        batch[:, 1:4] = random_flip_point_cloud(batch[:, 1:4])
        batch[:, 1:4] = random_scale_point_cloud(batch[:, 1:4])
        batch = self.subsample(batch)
        return batch

    def crop(self, batch):
        sample_point = batch[np.random.choice(range(len(batch))), 1:4]
        crop_x = np.random.normal(15, 2)
        crop_y = np.random.normal(15, 2)

        dist = torch.abs(batch[:, 1:4] - sample_point).reshape(-1, 3)
        mask = dist[:, 0] < crop_x
        mask = torch.logical_and(mask, dist[:, 1] < crop_y)
        return batch[mask]

    def subsample(self, batch, max_dropout_ratio=0.5):
        dropout = (1 - max_dropout_ratio) * torch.rand(1) + max_dropout_ratio
        keep = torch.rand(len(batch)) < dropout
        return batch[keep]
