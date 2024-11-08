#!/usr/bin/env python
# coding: utf-8

import glob
import os
import pathlib

import PIL
from torchvision import transforms

import wandb


def download_model(artifact: str, model_dir: str):
    """Download the model from Weights and Biases"""

    api = wandb.Api()
    artifact = api.artifact(artifact)
    artifact.download(root=model_dir)


def change_model_name(model_dir: pathlib.PosixPath, run_name: str):
    """Change the model name to the run name"""

    files = glob.glob(model_dir / "*")
    latest_file = max(files, key=os.path.getctime)
    new_model = model_dir / (run_name + ".pth")
    os.rename(latest_file, new_model)

    return new_model


def apply_transform_to_image(image: PIL.Image.Image):
    """Apply tensor transform to image"""

    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    image = transform_to_tensor(image)

    return image
