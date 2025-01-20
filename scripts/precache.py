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

from pathlib import Path
from typing import List, Optional

import typer
from tqdm import tqdm

from mos4d.config import load_config


def precache(
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
    sequence: List[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] Cache specific sequences",
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
    from torch.utils.data import DataLoader

    from mos4d.datasets.mos4d_dataset import MOS4DDataset as Dataset
    from mos4d.datasets.mos4d_dataset import collate_fn

    cfg = load_config(config)
    sequences = list(sequence) if sequence != None else cfg.training.train + cfg.training.val

    data_iterable = DataLoader(
        Dataset(
            dataloader=dataloader,
            data_dir=data,
            config=cfg,
            sequences=sequences,
            cache_dir=cache_dir,
        ),
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        batch_sampler=None,
    )

    for _ in tqdm(data_iterable, desc="Caching data", unit=" items", dynamic_ncols=True):
        pass


if __name__ == "__main__":
    import torch

    with torch.no_grad():
        typer.run(precache)
