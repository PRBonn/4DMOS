#!/usr/bin/env python3
# @file      confidences_to_labels.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import click
import os
import yaml
import copy
from tqdm import tqdm
import numpy as np
from mos4d.datasets.utils import load_files


@click.command()
### Add your options here
@click.option("--path", "-p", type=str, help="path to predictions.", required=True)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["non-overlapping", "bayes"]),
    help="strategy to predict from confidences.",
    default="non-overlapping",
)
@click.option(
    "--sequence",
    "-seq",
    type=int,
    help="Run inference on a specific sequence. Otherwise, test split from SemanticKITTI is used.",
    default=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    multiple=True,
)
@click.option(
    "--prior",
    "-prior",
    type=float,
    help="Moving prior for bayesian fusing.",
    default=0.5,
)
@click.option(
    "--dt",
    "-dt",
    type=float,
    help="Temporal resolution that is used for prediction.",
    default=0.1,
)
def main(path, strategy, sequence, prior, dt):
    sequences = list(sequence)

    semantic_config = yaml.safe_load(open("./config/semantic-kitti-mos.yaml"))
    strategy_str = strategy
    if strategy == "bayes":
        strategy_str = strategy_str + "_{:.3e}".format(float(prior))
    strategy_str = strategy_str + "_" + str(dt)

    for seq in tqdm(sequences, desc="Sequences"):
        pred_path = os.path.join(
            path,
            "labels",
            "strategy_" + strategy_str,
            "sequences",
            str(seq).zfill(2),
            "predictions",
        )
        os.makedirs(pred_path, exist_ok=True)

        path_to_sequence = os.path.join(path, "confidences", str(seq).zfill(2))
        current_indices = os.listdir(path_to_sequence)
        current_indices.sort()
        current = {}
        for current_idx in current_indices:
            path_to_current_idx = os.path.join(path_to_sequence, current_idx)
            pred_indices = os.listdir(path_to_current_idx)
            pred_indices.sort()
            current[current_idx] = pred_indices

        dict_confidences = {}
        for current_idx, predicted_confidences in current.items():
            for predicted_confidence in predicted_confidences:
                path_to_confidence = os.path.join(
                    path_to_sequence, current_idx, predicted_confidence
                )

                # Consider prediction if done with desired temporal resolution
                pred_idx = predicted_confidence.split("_")[0]
                temporal_resolution = float(predicted_confidence.split("_")[-1].split(".")[0])
                if temporal_resolution == dt:
                    if pred_idx not in dict_confidences:
                        dict_confidences[pred_idx] = [path_to_confidence]
                    else:
                        dict_confidences[pred_idx].append(path_to_confidence)
                    dict_confidences[pred_idx].sort()

        # Use all scans as prediction
        if strategy == "non-overlapping":
            for pred_idx, confidences in tqdm(dict_confidences.items(), desc="Scans"):
                from_idx = int(pred_idx) % len(confidences)
                confidence = np.load(confidences[from_idx])
                pred_labels = to_label(confidence, semantic_config)
                pred_labels.tofile(pred_path + "/" + pred_idx.split(".")[0] + ".label")

        # Bayesian Fusion
        elif strategy == "bayes":
            for pred_idx, confidences in tqdm(dict_confidences.items(), desc="Scans"):
                confidence = np.load(confidences[0])
                log_odds = prob_to_log_odds(confidence)
                for conf in confidences[1:]:
                    confidence = np.load(conf)
                    log_odds += prob_to_log_odds(confidence)
                    log_odds -= prob_to_log_odds(prior * np.ones_like(confidence))
                final_confidence = log_odds_to_prob(log_odds)
                pred_labels = to_label(final_confidence, semantic_config)
                pred_labels.tofile(pred_path + "/" + pred_idx.split(".")[0] + ".label")
        verify_predictions(seq, pred_path, semantic_config)


def verify_predictions(seq, pred_path, semantic_config):
    path_to_seq = os.path.join("/data", str(seq).zfill(2))
    scan_path = os.path.join(path_to_seq, "velodyne")
    filenames = load_files(scan_path)

    predicted_labels = os.listdir(pred_path)
    pred_scans = [name.split(".")[0] for name in predicted_labels]

    scans = [name.split(".")[0].split("/")[-1] for name in filenames]

    for i, scan in enumerate(scans):
        if scan not in pred_scans:
            pcd = np.fromfile(filenames[i], dtype=np.float32)
            pcd = pcd.reshape((-1, 4))
            n_points = pcd.shape[0]
            pred_labels = np.ones(n_points).astype(np.int32)
            pred_labels = to_original_labels(pred_labels, semantic_config)
            pred_labels.tofile(os.path.join(pred_path, scan + ".label"))
            print("Created artificial label for scan {}".format(scan))


def to_original_labels(labels, semantic_config):
    original_labels = copy.deepcopy(labels)
    for k, v in semantic_config["learning_map_inv"].items():
        original_labels[labels == k] = v
    return original_labels


def to_label(confidence, semantic_config):
    pred_labels = np.ones_like(confidence)
    pred_labels[confidence > 0.5] = 2
    pred_labels = to_original_labels(pred_labels, semantic_config)
    pred_labels = pred_labels.reshape((-1)).astype(np.int32)
    return pred_labels


def prob_to_log_odds(prob):
    odds = np.divide(prob, 1 - prob + 1e-10)
    log_odds = np.log(odds)
    return log_odds


def log_odds_to_prob(log_odds):
    log_odds = np.clip(log_odds, -80, 80)
    odds = np.exp(log_odds)
    prob = np.divide(odds, odds + 1)
    return prob


if __name__ == "__main__":
    main()
