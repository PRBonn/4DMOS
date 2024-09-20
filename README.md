<div align="center">
  <h1>Receding Moving Object Segmentation in 3D LiDAR Data Using Sparse 4D Convolutions</h1>
  <a href="https://github.com/PRBonn/4DMOS#how-to-use-it"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/4DMOS#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.ipb.uni-bonn.de/pdfs/mersch2022ral.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
</div>

![example](docs/4dmos.gif)
*Our moving object segmentation on the unseen SemanticKITTI test sequences 18 and 21. Red points are predicted as moving.*

Please find the corresponding video [here](https://youtu.be/5aWew6caPNQ).

<p align="center">
    <img src="docs/introduction.png" width="600">
</p>

*Given a sequence of point clouds, our method segments moving (red) from non-moving (black) points.*

<p align="center">
    <img src="docs/architecture.png">
</p>

*We first create a sparse 4D point cloud of all points in a given receding window. We use sparse 4D convolutions from the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) to extract spatio-temporal features and predict per-points moving object scores.*

## Important Update
The current state of the repository is improved by internally aligning the scans using [KISS-ICP](https://github.com/PRBonn/kiss-icp). Also, the build system and pipeline are inspired from [MapMOS](https://github.com/PRBonn/MapMOS), so you can run it on most point cloud data formats. If you want to reproduce the original results from the paper, this version is tagged under `0.1`. You can checkout by

```bash
git checkout v0.1
```

## Installation
First, make sure the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) is installed on your system, see [here](https://github.com/NVIDIA/MinkowskiEngine#installation) for more details.

Next, clone our repository
```bash
git clone git@github.com:PRBonn/4DMOS && cd 4DMOS
```

and install with
```bash
make install
```

**or**
```bash
make install-all
```
if you want to install the project with all optional dependencies (needed for the visualizer). In case you want to edit the Python code, install in editable mode:
```bash
make editable
```

## How to Use It
Just type

```bash
mos4d_pipeline --help
```
to see how to run 4DMOS.

Check the [Download](#downloads) section for a pre-trained model. Like [KISS-ICP](https://github.com/PRBonn/kiss-icp), our pipeline runs on a variety of point cloud data formats like `bin`, `pcd`, `ply`, `xyz`, `rosbags`, and more. To visualize these, just type

```bash
mos4d_pipeline --visualize /path/to/weights.ckpt /path/to/data
```

<details>
<summary>Want to evaluate with ground truth labels?</summary>

Because these labels come in all shapes, you need to specify a dataloader. This is currently available for SemanticKITTI, NuScenes, HeLiMOS, and our labeled KITTI Tracking sequence 19 and Apollo sequences (see [Downloads](#downloads)).

</details>

## Training
To train our approach, you need to first cache your data. To see how to do that, just `cd` into the `4DMOS` repository and type

```bash
python3 scripts/precache.py --help
```

After this, you can run the training script. Again, `--help` shows you how:
```bash
python3 scripts/train.py --help
```

<details>
<summary>Want to verify the cached data?</summary>

You can inspect the cached training samples by using the script `python3 scripts/cache_to_ply.py --help`.

</details>

<details>
<summary>Want to change the logging directory?</summary>

The training log and checkpoints will be saved by default to the current working directory. To change that, export the `export LOGS=/your/path/to/logs` environment variable before running the training script.

</details>

## HeLiMOS
To train on the HeLiMOS data with different sensor configurations, use the following commands:

```shell
python3 scripts/precache.py /path/to/HeLiMOS helimos /path/to/cache --config config/helimos/*_training.yaml
python3 scripts/train.py /path/to/HeLiMOS helimos /path/to/cache --config config/helimos/*_training.yaml
```

by replacing the paths and the config file names. To evaluate for example on the Velodyne test data, run

```shell
mos4d_pipeline /path/to/weights.ckpt /path/to/HeLiMOS --dataloader helimos -s Velodyne/test.txt
```

## Evaluation and Visualization
We use the [SemanticKITTI API](https://github.com/PRBonn/semantic-kitti-api) to evaluate the intersection-over-union (IOU) of the moving class as well as to visualize the predictions. Clone the repository in your workspace, install the dependencies and then run the following command to visualize your predictions for e.g. sequence 8:

```
cd semantic-kitti-api
./visualize_mos.py --sequence 8 --dataset /path/to/dataset --predictions /path/to/4DMOS/predictions/ID/POSES/labels/STRATEGY/
```

## Benchmark
To submit the results to the LiDAR-MOS benchmark, please follow the instructions [here](https://competitions.codalab.org/competitions/28894).

## Downloads
<p align="center">
    <img src="docs/table.png" width="600">
</p>

* [Model [A]: 5 scans @ 0.1s](https://www.ipb.uni-bonn.de/html/projects/4DMOS/5_scans.zip)
* [Model [B]: 5 scans @ 0.2s](https://www.ipb.uni-bonn.de/html/projects/4DMOS/5_scans_dt_0p2.zip)
* [Model [C]: 5 scans @ 0.3s](https://www.ipb.uni-bonn.de/html/projects/4DMOS/5_scans_dt_0p3.zip)
* [Model [D]: 5 scans, no poses](https://www.ipb.uni-bonn.de/html/projects/4DMOS/5_scans_no_poses.zip)
* [Model [E]: 5 scans input, 1 scan output](https://www.ipb.uni-bonn.de/html/projects/4DMOS/5_scans_single_output.zip)
* [Model [F]: 2 scans](https://www.ipb.uni-bonn.de/html/projects/4DMOS/2_scans.zip)
* [Model [G]: 10 scans](https://www.ipb.uni-bonn.de/html/projects/4DMOS/10_scans.zip)

## Publication
If you use our code in your academic work, please cite the corresponding [paper](https://www.ipb.uni-bonn.de/pdfs/mersch2022ral.pdf):

```bibtex
@article{mersch2022ral,
author = {B. Mersch and X. Chen and I. Vizzo and L. Nunes and J. Behley and C. Stachniss},
title = {{Receding Moving Object Segmentation in 3D LiDAR Data Using Sparse 4D Convolutions}},
journal={IEEE Robotics and Automation Letters (RA-L)},
year = 2022,
volume = {7},
number = {3},
pages = {7503--7510},
codeurl = {https://github.com/PRBonn/4DMOS},
}
```

## Acknowledgments
This implementation is heavily inspired by [KISS-ICP](https://github.com/PRBonn/kiss-icp).

## License
This project is free software made available under the MIT License. For details see the LICENSE file.
