"""
Author: Aditya Jain
Date started: April 18, 2024
About: Evaluation of AMI-GBIF trained binary classifier on AMI-Traps data
"""

import argparse
import glob
import json
import os
import pathlib
from pathlib import Path

import torch
from model_inference import ModelInference
from PIL import Image

from .helper_functions import (
    apply_transform_to_image,
    change_model_name,
    download_model,
)


def _get_insect_crops_and_labels(insect_crops_dir: pathlib.PosixPath):
    """Get all insect crops and label information"""

    insect_crops = glob.glob(insect_crops_dir / "*.jpg")
    with open(insect_crops_dir / "binary_labels.json") as f:
        insect_labels = json.load(f)

    return insect_crops, insect_labels


def _update_evaluation_metrics(
    gt_label: str,
    pred: str,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    gt_moths: int,
    gt_nonmoths: int,
):
    """Update evaluation metrics"""

    if gt_label == "Moth":
        if pred == "moth":
            tp += 1
        else:
            fn += 1
        gt_moths += 1
    elif gt_label == "Non-Moth":
        if pred == "nonmoth":
            tn += 1
        else:
            fp += 1
        gt_nonmoths += 1
    else:
        raise Exception("Unknown binary label for an insect crop!")

    return tp, tn, fp, fn, gt_moths, gt_nonmoths


def _get_aggregated_metrics(
    tp: int, tn: int, fp: int, fn: int, gt_moths: int, gt_nonmoths: int
):
    """Get aggregated metrics"""

    total_crops = gt_moths + gt_nonmoths
    accuracy = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)
    precision = round((tp) / (tp + fp) * 100, 2)
    recall = round((tp) / (tp + fn) * 100, 2)
    fscore = round((2 * precision * recall) / (precision + recall), 2)

    return total_crops, accuracy, precision, recall, fscore


def binary_model_evaluation(
    run_name: str,
    artifact: str,
    model_type: str,
    model_dir: str,
    category_map: str,
    insect_crops_dir: str,
    skip_small_crops: bool,
    min_crop_dim: int = 0,
):
    """Main function for binary model evaluation"""

    # Get the environment variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device} is available.")
    model_dir = Path(model_dir)
    insect_crops_dir = Path(insect_crops_dir)

    # Download the model from Weights and Biases
    download_model(artifact, model_dir)

    # Change the downloaded model name to the run name
    new_model = change_model_name(model_dir, run_name)

    # Build the binary classification model
    categ_map_path = model_dir / category_map
    binary_classifier = ModelInference(new_model, model_type, categ_map_path, device)

    # Get all insect crops and label information
    insect_crops, insect_labels = _get_insect_crops_and_labels(insect_crops_dir)

    # Evaluation metrics variables
    tp, tn, fp, fn = 0, 0, 0, 0
    gt_moths, gt_nonmoths = 0, 0

    # Iterate over the files in webdataset
    for image_path in insect_crops:
        # Read the image
        image = Image.open(image_path)

        # If skipping small crops, check size
        if skip_small_crops == "True":
            width, height = image.width, image.height
            if width < min_crop_dim and height < min_crop_dim:
                continue

        # Get ground truth label
        img_name = os.path.split(image_path)[1]
        gt_label = insect_labels[img_name]["label"]

        # Binary model prediction
        image = apply_transform_to_image(image)
        pred = binary_classifier.predict(image)[0][0]

        # Fill up evluation metrics
        tp, tn, fp, fn, gt_moths, gt_nonmoths = _update_evaluation_metrics(
            gt_label, pred, tp, tn, fp, fn, gt_moths, gt_nonmoths
        )

    # Aggregated metrics
    total_crops, accuracy, precision, recall, fscore = _get_aggregated_metrics(
        tp, tn, fp, fn, gt_moths, gt_nonmoths
    )

    print(
        f"\nBinary classification evaluation for {run_name}:\
        \nSkip small crops - {skip_small_crops}; Min. crop dimension - {min_crop_dim}\
        \nTotal insect crops - {total_crops}\
        \nGT moth crops - {gt_moths} ({round(gt_moths/total_crops*100,2)}%)\
        \nGT non-moth crops - {gt_nonmoths} ({round(gt_nonmoths/total_crops*100,2)}%)\
        \nAccuracy - {accuracy}%\
        \nPrecision - {precision}%\
        \nRecall - {recall}%\
        \nF1 score - {fscore}%\
        \n",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        help="Run name of the model on Weights and Biases.",
        required=True,
    )
    parser.add_argument(
        "--wandb_model_artifact",
        help="Model artifact on Weights and Biases.",
        required=True,
    )
    parser.add_argument(
        "--model_type",
        help="Model type of the binary classifier.",
        required=True,
    )
    parser.add_argument(
        "--model_dir",
        help="Model directory where the binary models are downloaded.",
        required=True,
    )
    parser.add_argument(
        "--category_map",
        help="Category map for the binary classifier.",
        required=True,
    )
    parser.add_argument(
        "--insect_crops_dir",
        help="Directory containing the insect crops.",
        required=True,
    )
    parser.add_argument(
        "--skip_small_crops",
        help="Whether to skip crops below a certain size.",
        required=True,
    )
    parser.add_argument(
        "--min_crop_dim",
        help="Minimum crop length in pixels to consider for prediction.",
        type=int,
    )

    args = parser.parse_args()
    binary_model_evaluation(
        args.run_name,
        args.wandb_model_artifact,
        args.model_type,
        args.model_dir,
        args.category_map,
        args.insect_crops_dir,
        args.skip_small_crops,
        args.min_crop_dim,
    )
