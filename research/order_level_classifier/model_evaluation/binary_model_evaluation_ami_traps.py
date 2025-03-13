"""
Author: Aditya Jain
Date started: April 18, 2024
About: Evaluation of AMI-GBIF trained binary classifier on AMI-Traps data
"""

import glob
import json
import os
import pathlib
from pathlib import Path

import torch

# 3rd party packages
from dotenv import load_dotenv
from PIL import Image

from src.classification.model_inference import ModelInference

# Load secrets and config from optional .env file
load_dotenv()


def _get_insect_crops_and_labels(insect_crops_dir: pathlib.PosixPath):
    """Get all insect crops and label information"""

    insect_crops = glob.glob(str(insect_crops_dir / "*.png"))
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
        if pred == "Moth":
            tp += 1
        else:
            fn += 1
        gt_moths += 1
    elif gt_label == "Non-Moth":
        if pred == "Non-Moth":
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
    model: str,
    model_type: str,
    model_dir: str,
    category_map_key_to_id: str,
    category_map_key_to_name: str,
    insect_crops_dir: str,
    skip_small_crops: bool = False,
    min_crop_dim: int = 0,
):
    """Main function for binary model evaluation"""
    print(f"Evaluating model for the run {run_name}.")

    # Get the environment variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device} is available.")
    model_checkpoint_path = str(Path(model_dir) / model)
    category_map_key_to_id_path = str(Path(model_dir) / category_map_key_to_id)
    category_map_key_to_name_path = str(Path(model_dir) / category_map_key_to_name)
    insect_crops_dir = Path(insect_crops_dir)

    # Build the binary classification model
    order_classifier = ModelInference(
        model_checkpoint_path,
        model_type,
        category_map_key_to_id_path,
        category_map_key_to_name_path,
        device,
        checkpoint=True,
        topk=1,
    )

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
        if skip_small_crops:
            width, height = image.width, image.height
            if width < min_crop_dim and height < min_crop_dim:
                continue

        # Get ground truth label
        img_name = os.path.split(image_path)[1]
        gt_label = insect_labels[img_name]["label"]

        # Binary model prediction
        # image = _apply_transform_to_image(image)
        prediction = order_classifier.predict(image)
        predicted_label = prediction[0][0]
        if predicted_label == "Lepidoptera":
            binary_prediction = "Moth"
        else:
            binary_prediction = "Non-Moth"

        # Fill up evluation metrics
        tp, tn, fp, fn, gt_moths, gt_nonmoths = _update_evaluation_metrics(
            gt_label, binary_prediction, tp, tn, fp, fn, gt_moths, gt_nonmoths
        )

    # Aggregated metrics
    total_crops, accuracy, precision, recall, fscore = _get_aggregated_metrics(
        tp, tn, fp, fn, gt_moths, gt_nonmoths
    )

    print(
        f"\nBinary classification evaluation for {model}:\
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
    ORDER_MODEL_DIR = os.getenv("ORDER_MODEL_DIR")
    ORDER_INSECT_CROPS_DIR = os.getenv("ORDER_INSECT_CROPS_DIR")

    binary_model_evaluation(
        run_name="worder0.2_wbinary0.8_run2",
        model="convnext_tiny_in22k_20250117_033404_checkpoint.pt",
        model_type="convnext_tiny_in22k",
        model_dir=ORDER_MODEL_DIR,
        category_map_key_to_id="taxon_key_to_id_map.json",
        category_map_key_to_name="taxon_key_to_name_map.json",
        insect_crops_dir=ORDER_INSECT_CROPS_DIR,
    )
