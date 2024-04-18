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

    # Build the webdataset for binary classification dataset
    transform_img = transforms.Compose([transforms.ToTensor()])
    data = (wds.WebDataset(wbds_files)
           .decode("pil")
           .to_tuple("jpg", "json")
           .map_tuple(transform_img, identity))
    dataloader = torch.utils.data.DataLoader(data)

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

    # Iterate over the files in webdataset
    for image, annotation in dataloader:
        pass


if __name__ == "__main__":
    run_name = "binary_resnet50_baseline_run3"
    wandb_model_artifact = "moth-ai/ami-gbif-binary/model:v1"
    model_type = "resnet50"
    model_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/models/binary"
    category_map = "05-moth-nonmoth_category_map.json"
    wbds_files = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/webdataset/binary_classification/binary-{000000..000011}.tar"
    
    binary_model_evaluation(
        run_name, wandb_model_artifact, model_type, model_dir, category_map, wbds_files
    )
