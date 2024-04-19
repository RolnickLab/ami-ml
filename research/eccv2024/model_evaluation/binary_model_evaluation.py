"""
Author: Aditya Jain
Date started: April 18, 2024
About: Evaluation of AMI-GBIF trained binary classifier on AMI-Traps data
"""

import os
import glob
import json
import argparse
from PIL import Image
import torch
from torchvision import transforms
import wandb

from model_inference import ModelInference


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

    # Download the model
    api = wandb.Api()
    artifact = api.artifact(artifact)
    artifact.download(root=model_dir)

    # Change downloaded model name to the run name
    files = glob.glob(os.path.join(model_dir, "*"))
    latest_file = max(files, key=os.path.getctime)
    new_model = os.path.join(model_dir, run_name + ".pth")
    os.rename(latest_file, new_model)

    # Build the binary classification model
    categ_map_path = os.path.join(model_dir, category_map)
    binary_classifier = ModelInference(new_model, model_type, categ_map_path, device)

    # Get all insect crops and label information
    insect_crops = glob.glob(os.path.join(insect_crops_dir, "*.jpg"))
    insect_labels = json.load(
        open(os.path.join(insect_crops_dir, "binary_labels.json"))
    )

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
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        image = transform_to_tensor(image)
        pred = binary_classifier.predict(image)[0][0]

        # Fill up evluation metrics
        if gt_label == "Moth":
            if pred == "moth": tp += 1
            else: fn += 1
            gt_moths += 1
        elif gt_label == "Non-Moth":
            if pred == "nonmoth": tn += 1
            else: fp += 1
            gt_nonmoths += 1
        else:
            raise Exception("Unknown binary label for an insect crop!")

    # Aggregated metrics
    total_crops = gt_moths + gt_nonmoths
    accuracy = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)
    precision = round((tp) / (tp + fp) * 100, 2)
    recall = round((tp) / (tp + fn) * 100, 2)
    fscore = round((2 * precision * recall) / (precision + recall), 2)

    print(
        f"\nBinary classification evaluation for {run_name}:\
        \nSkip small crops - {skip_small_crops}; Min. crop dimension - {min_crop_dim}\
        \nTotal insect crops - {total_crops}\
        \nGround-truth moth crops - {gt_moths} ({round(gt_moths/total_crops*100,2)}%)\
        \nGround-truth non-moth crops - {gt_nonmoths} ({round(gt_nonmoths/total_crops*100,2)}%)\
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
        bool(args.skip_small_crops),
        args.min_crop_dim,
    )
