#!/usr/bin/env python
# coding: utf-8

"""Utility functions for confidence calibration"""

import glob
import json
import pathlib

import PIL
import torch
import torch.nn as nn
from torchvision import transforms


class TemperatureScaling(nn.Module):
    """Temperature scaling class"""

    def __init__(self, initial_t: float = 1.0):
        super().__init__()
        print(f"The initial value of temp scaling is {initial_t}.")
        self.temperature = nn.Parameter(torch.ones(1) * initial_t)  # Initialize T = 1

    def forward(self, logits):
        return logits / self.temperature


def get_insect_crops_and_labels(insect_crops_dir: pathlib.PosixPath):
    """Get all insect crops and label information"""

    insect_crops = glob.iglob(str(insect_crops_dir / "*.png"))
    with open(insect_crops_dir / "fgrained_labels.json") as f:
        insect_labels = json.load(f)

    return insect_crops, insect_labels


def get_category_map(category_map_file: str):
    """Read and return the model category map"""

    with open(category_map_file) as f:
        category_map = json.load(f)

    return category_map


def pad_to_square(width: int, height: int):
    """Padding transformation to make the image square"""

    if height < width:
        return transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        return transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        return transforms.Pad(padding=[0, 0, 0, 0])


def apply_transform_to_image(
    image: PIL.Image.Image, input_size: int = 128, preprocess_mode: str = "torch"
):
    """Apply tensor transform to image"""

    width, height = image.size
    padding_transform = pad_to_square(width, height)

    preprocess_params = {
        "torch": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "tf": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "default": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    }

    mean, std = preprocess_params.get(preprocess_mode, preprocess_params["default"])

    image_transform = transforms.Compose(
        [
            padding_transform,
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    image = image_transform(image)

    return image


def binary_search():
    """Binary search on the temperature value T"""
