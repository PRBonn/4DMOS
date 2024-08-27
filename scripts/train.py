#!/usr/bin/env python3
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
import torch
import typer
from typing import Optional
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from mos4d.utils.seed import set_seed
from mos4d.config import load_config


def train(
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    dataloader: str = typer.Argument(
        ...,
        help="The dataloader to be used",
        show_default=False,
    ),
    cache_dir: Path = typer.Argument(
        ...,
        help="The directory where the cache should be created",
        show_default=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
):
    from mos4d.datasets.mos4d_dataset import MOS4DDataModule as DataModule
    from mos4d.training_module import TrainingModule

    set_seed(66)

    cfg = load_config(config)
    model = TrainingModule(cfg)

    # Load data and model
    dataset = DataModule(
        dataloader=dataloader,
        data_dir=data,
        config=cfg,
        cache_dir=cache_dir,
    )

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="val_moving_iou",
        filename=cfg.training.id + "_{epoch:03d}_{val_moving_iou:.3f}",
        mode="max",
        save_last=True,
    )

    # Logger
    environ = os.environ.get("LOGS")
    prefix = environ if environ is not None else "models"
    log_dir = os.path.join(prefix, "4DMOS")
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        log_dir,
        name=cfg.training.id,
        default_hp_metric=False,
    )

    torch.set_float32_matmul_precision("high")
    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=tb_logger,
        max_epochs=cfg.training.max_epochs,
        callbacks=[
            lr_monitor,
            checkpoint_saver,
        ],
    )

    # Train!
    trainer.fit(model, dataset)


if __name__ == "__main__":
    typer.run(train)
