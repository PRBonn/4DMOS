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
import typer
import importlib

import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from mos4d.datasets.mos4d_dataset import MOS4DDataset, collate_fn
from mos4d.config import load_config


def cache_to_ply(
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
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
):
    try:
        o3d = importlib.import_module("open3d")
    except ModuleNotFoundError as err:
        print(f'open3d is not installed on your system, run "pip install open3d"')
        exit(1)

    # Run
    cfg = load_config(config)

    data_iterable = DataLoader(
        MOS4DDataset(
            dataloader=dataloader,
            data_dir=data,
            config=cfg,
            sequences=[sequence],
            cache_dir=cache_dir,
        ),
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        batch_sampler=None,
    )

    dataset_sequence = (
        data_iterable.dataset.datasets[sequence].sequence_id
        if hasattr(data_iterable.dataset.datasets[sequence], "sequence_id")
        else os.path.basename(data_iterable.dataset.datasets[seq].data_dir)
    )
    path = os.path.join("ply", dataset_sequence)
    os.makedirs(path, exist_ok=True)

    for idx, batch in enumerate(
        tqdm(data_iterable, desc="Writing data to ply", unit=" items", dynamic_ncols=True)
    ):
        if len(batch) > 0:
            indices = torch.unique(batch[:, 4])
            final_pcd = o3d.geometry.PointCloud()
            for scan_index in indices:
                color_value = (1 + scan_index - torch.min(indices)) / (
                    1 + torch.max(indices - torch.min(indices))
                )
                mask = batch[:, 4] == scan_index
                pcd = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(batch[mask, 1:4].numpy())
                ).paint_uniform_color([color_value, color_value, color_value])
                colors = np.array(pcd.colors)
                colors[batch[mask, -1] == 1.0] = [1, 0, 0]
                pcd.colors = o3d.utility.Vector3dVector(colors)
                final_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, f"{idx:06}.ply"), final_pcd)


if __name__ == "__main__":
    typer.run(cache_to_ply)
