"""
Author: Aditya Jain
Date started: April 18, 2024
About: Evaluation of AMI-GBIF trained binary classifier on AMI-Traps data
"""

import os
import glob
import webdataset as wds
from itertools import islice
import pandas as pd
import argparse
from PIL import Image
import torch
from torchvision import transforms
import wandb

from model_inference import ModelInference


def identity(x):
    return x


def binary_model_evaluation(
    run_name: str,
    artifact: str,
    model_type: str,
    model_dir: str,
    category_map: str,
    wbds_files: str,
):
    """Main function for binary model evaluation"""

    # Get the environment variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device} is available.")

    # Build the webdataset for binary classification dataset
    transform_img = transforms.Compose([transforms.ToTensor()])
    data = (wds.WebDataset(wbds_files)
           .decode("pil")
           .to_tuple("jpg", "json")
           .map_tuple(transform_img, identity))
    dataloader = torch.utils.data.DataLoader(data, num_workers=1)

    # Download the model
    api = wandb.Api()
    artifact = api.artifact(artifact)
    artifact.download(root=model_dir)

    # Change downloaded model name to the run name
    files = glob.glob(os.path.join(model_dir, "*"))
    latest_file = max(files, key=os.path.getctime)
    new_model = os.path.join(model_dir, run_name + ".pth")
    os.rename(latest_file, new_model)

    new_model = os.path.join(model_dir, "moth-nonmoth_resnet50_20230604_065440_30.pth") # TEST

    # Build the binary classification model
    categ_map_path = os.path.join(model_dir, category_map)
    binary_classifier = ModelInference(new_model, model_type, categ_map_path, device)

    # Evaluation metrics variables
    tp, tn, fp, fn = 0, 0, 0, 0
    gt_moths, gt_nonmoths = 0, 0

    # Iterate over the files in webdataset
    for image, annotation in dataloader: 

        image = image.to(
                device, non_blocking=True
        )

        # Get ground truth label 
        gt_label = annotation["label"][0]

        # Binary model prediction
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
            raise Exception("Unknown binary label for a moth crop!")
        
    # Aggregated metrics
    total_crops = gt_moths + gt_nonmoths
    accuracy = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)
    precision = round((tp) / (tp + fp) * 100, 2)
    recall = round((tp) / (tp + fn) * 100, 2)
    fscore = round((2 * precision * recall) / (precision + recall), 2)

    print(
        f"\nBinary classification evaluation for {run_name}:\
        \nTotal insect crops - {total_crops}\
        \nGround-truth moth crops - {gt_moths} ({round(gt_moths/total_crops*100,2)}%)\
        \nGround-truth non-moth crops - {gt_nonmoths} ({round(gt_nonmoths/total_crops*100,2)}%)\
        \nAccuracy - {accuracy}%\
        \nPrecision - {precision}%\
        \nRecall - {recall}%\
        \nF1 score - {fscore}%\
        \n"
    )

if __name__ == "__main__":
    run_name = "binary_resnet50_baseline_run1"
    wandb_model_artifact = "moth-ai/ami-gbif-binary/model:v0"
    model_type = "resnet50"
    model_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/models/binary"
    category_map = "05-moth-nonmoth_category_map.json"
    wbds_files = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/webdataset/binary_classification/binary-{000000..000011}.tar"
    skip_small_crops = True
    min_crop_dim = 100

    binary_model_evaluation(
        run_name, wandb_model_artifact, model_type, model_dir, category_map, wbds_files
    )
