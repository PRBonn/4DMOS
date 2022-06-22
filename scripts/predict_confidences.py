#!/usr/bin/env python3
# @file      predict_confidences.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import click
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F

import mos4d.datasets.datasets as datasets
import mos4d.models.models as models


@click.command()
### Add your options here
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to checkpoint file (.ckpt) to do inference.",
    required=True,
)
@click.option(
    "--sequence",
    "-seq",
    type=int,
    help="Run inference on a specific sequence. Otherwise, test split from config is used.",
    default=None,
    multiple=True,
)
@click.option(
    "--dt",
    "-dt",
    type=float,
    help="Desired temporal resolution of predictions.",
    default=None,
)
@click.option(
    "--poses", "-poses", type=str, default=None, help="Specify which poses to use."
)
@click.option(
    "--transform",
    "-transform",
    type=bool,
    default=None,
    help="Transform point clouds to common viewpoint.",
)
def main(weights, sequence, dt, poses, transform):
    cfg = torch.load(weights)["hyper_parameters"]

    if poses:
        cfg["DATA"]["POSES"] = poses

    if transform != None:
        cfg["DATA"]["TRANSFORM"] = transform
        if not transform:
            cfg["DATA"]["POSES"] = "no_poses"

    if sequence:
        cfg["DATA"]["SPLIT"]["TEST"] = list(sequence)

    if dt:
        cfg["MODEL"]["DELTA_T_PREDICTION"] = dt

    cfg["TRAIN"]["BATCH_SIZE"] = 1

    # Load data and model
    cfg["DATA"]["SPLIT"]["TRAIN"] = cfg["DATA"]["SPLIT"]["TEST"]
    cfg["DATA"]["SPLIT"]["VAL"] = cfg["DATA"]["SPLIT"]["TEST"]
    data = datasets.KittiSequentialModule(cfg)
    data.setup()

    model = models.MOSNet.load_from_checkpoint(weights, hparams=cfg)

    # Setup trainer
    trainer = Trainer(gpus=1, logger=False)

    # Infer!
    trainer.predict(model, data.test_dataloader())


if __name__ == "__main__":
    main()
