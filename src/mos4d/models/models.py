#!/usr/bin/env python3
# @file      models.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.core.lightning import LightningModule
import MinkowskiEngine as ME

from mos4d.models.MinkowskiEngine.customminkunet import CustomMinkUNet
from mos4d.models.loss import MOSLoss
from mos4d.models.metrics import ClassificationMetrics


class MOSNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.poses = (
            self.hparams["DATA"]["POSES"].split(".")[0]
            if self.hparams["DATA"]["TRANSFORM"]
            else "no_poses"
        )
        self.id = self.hparams["EXPERIMENT"]["ID"]
        self.dt_prediction = self.hparams["MODEL"]["DELTA_T_PREDICTION"]
        self.lr = self.hparams["TRAIN"]["LR"]
        self.lr_epoch = hparams["TRAIN"]["LR_EPOCH"]
        self.lr_decay = hparams["TRAIN"]["LR_DECAY"]
        self.weight_decay = hparams["TRAIN"]["WEIGHT_DECAY"]
        self.n_past_steps = hparams["MODEL"]["N_PAST_STEPS"]

        self.semantic_config = yaml.safe_load(open(hparams["DATA"]["SEMANTIC_CONFIG_FILE"]))
        self.n_classes = len(self.semantic_config["learning_map_inv"])
        self.ignore_index = [
            key for key, ignore in self.semantic_config["learning_ignore"].items() if ignore
        ]
        self.model = MOSModel(hparams, self.n_classes)

        self.MOSLoss = MOSLoss(self.n_classes, self.ignore_index)

        self.ClassificationMetrics = ClassificationMetrics(self.n_classes, self.ignore_index)

    def getLoss(self, out: ME.TensorField, past_labels: list):
        loss = self.MOSLoss.compute_loss(out, past_labels)
        return loss

    def forward(self, past_point_clouds: dict):
        out = self.model(past_point_clouds)
        return out

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        _, past_point_clouds, past_labels = batch

        out = self.forward(past_point_clouds)

        loss = self.getLoss(out, past_labels)
        self.log("train_loss", loss.item(), on_step=True)

        # Logging metrics
        dict_confusion_matrix = {}
        for s in range(self.n_past_steps):
            dict_confusion_matrix[s] = (
                self.get_step_confusion_matrix(out, past_labels, s).detach().cpu()
            )

        torch.cuda.empty_cache()
        return {"loss": loss, "dict_confusion_matrix": dict_confusion_matrix}

    def training_epoch_end(self, training_step_outputs):
        list_dict_confusion_matrix = [
            output["dict_confusion_matrix"] for output in training_step_outputs
        ]
        for s in range(self.n_past_steps):
            agg_confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
            for dict_confusion_matrix in list_dict_confusion_matrix:
                agg_confusion_matrix = agg_confusion_matrix.add(dict_confusion_matrix[s])
            iou = self.ClassificationMetrics.getIoU(agg_confusion_matrix)
            self.log("train_moving_iou_step{}".format(s), iou[2].item())

        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):
        batch_size = len(batch[0])
        meta, past_point_clouds, past_labels = batch

        out = self.forward(past_point_clouds)

        loss = self.getLoss(out, past_labels)
        self.log("val_loss", loss.item(), batch_size=batch_size, prog_bar=True, on_epoch=True)

        dict_confusion_matrix = {}
        for s in range(self.n_past_steps):
            dict_confusion_matrix[s] = (
                self.get_step_confusion_matrix(out, past_labels, s).detach().cpu()
            )

        torch.cuda.empty_cache()
        return dict_confusion_matrix

    def validation_epoch_end(self, validation_step_outputs):
        for s in range(self.n_past_steps):
            agg_confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
            for dict_confusion_matrix in validation_step_outputs:
                agg_confusion_matrix = agg_confusion_matrix.add(dict_confusion_matrix[s])
            iou = self.ClassificationMetrics.getIoU(agg_confusion_matrix)
            self.log("val_moving_iou_step{}".format(s), iou[2].item())

        torch.cuda.empty_cache()

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = None):
        # torch.set_grad_enabled(True)
        meta, past_point_clouds, past_labels = batch
        out = self.forward(past_point_clouds)

        for b in range(len(batch[0])):
            seq, idx, past_indices = meta[b]
            path = os.path.join(
                "predictions",
                self.id,
                self.poses,
                "confidences",
                str(seq).zfill(2),
                str(idx).zfill(6),
            )
            os.makedirs(path, exist_ok=True)
            for step in range(self.n_past_steps):
                coords = out.coordinates_at(b)
                logits = out.features_at(b)

                t = round(-step * self.dt_prediction, 3)
                mask = coords[:, -1].isclose(torch.tensor(t))
                masked_logits = logits[mask]

                masked_logits[:, self.ignore_index] = -float("inf")

                pred_softmax = F.softmax(masked_logits, dim=1)
                pred_softmax = pred_softmax.detach().cpu().numpy()
                assert pred_softmax.shape[1] == 3
                assert pred_softmax.shape[0] >= 0
                sum = np.sum(pred_softmax[:, 1:3], axis=1)
                assert np.isclose(sum, np.ones_like(sum)).all()
                moving_confidence = pred_softmax[:, 2]

                file_name = os.path.join(
                    path,
                    str(past_indices[-step - 1]).zfill(6)
                    + "_dt_{:.0e}".format(self.dt_prediction)
                    + ".npy",
                )

                np.save(file_name, moving_confidence)

        torch.cuda.empty_cache()

    def get_step_confusion_matrix(self, out, past_labels, step):
        t = round(-step * self.dt_prediction, 3)
        mask = out.coordinates[:, -1].isclose(torch.tensor(t))
        pred_logits = out.features[mask].detach().cpu()
        gt_labels = torch.cat(past_labels, dim=0).detach().cpu()
        gt_labels = gt_labels[mask][:, 0]
        confusion_matrix = self.ClassificationMetrics.compute_confusion_matrix(
            pred_logits, gt_labels
        )
        return confusion_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_epoch, gamma=self.lr_decay
        )
        return [optimizer], [scheduler]


#######################################
# Modules
#######################################


class MOSModel(nn.Module):
    def __init__(self, cfg: dict, n_classes: int):
        super().__init__()
        self.dt_prediction = cfg["MODEL"]["DELTA_T_PREDICTION"]
        ds = cfg["DATA"]["VOXEL_SIZE"]
        self.quantization = torch.Tensor([ds, ds, ds, self.dt_prediction])
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=n_classes, D=4)

    def forward(self, past_point_clouds):
        quantization = self.quantization.type_as(past_point_clouds[0])

        past_point_clouds = [
            torch.div(point_cloud, quantization) for point_cloud in past_point_clouds
        ]
        features = [
            0.5 * torch.ones(len(point_cloud), 1).type_as(point_cloud)
            for point_cloud in past_point_clouds
        ]
        coords, features = ME.utils.sparse_collate(past_point_clouds, features)
        tensor_field = ME.TensorField(features=features, coordinates=coords.type_as(features))

        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)

        out = predicted_sparse_tensor.slice(tensor_field)
        out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
        return out
