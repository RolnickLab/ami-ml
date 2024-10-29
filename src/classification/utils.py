#!/usr/bin/env python
# coding: utf-8

""" Utility functions
"""

import os
import random
import typing as tp

import numpy as np

# import timm
import torch

from src.classification.models import model_list

# import webdataset as wds


def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility"""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


SupportedModels = tp.Literal[
    "efficientnetv2-b3",
    "efficientnetv2-s-in21k",
    "swin-s",
    "resnet50",
    "timm_mobilenetv3large",
    "timm_resnet50",
    "timm_convnext-t",
    "timm_convnext-b",
    "timm_vit-b16-128",
    "timm_vit-b16-224",
    "timm_vit-b16-384",
]


def model_builder(
    device: str,
    model_type: str,
    num_classes: int,
    existing_weights: tp.Optional[str],
    pretrained: bool = True,
):
    """Model builder"""

    model = model_list(model_type, num_classes, pretrained)

    # If available, load existing weights
    if existing_weights:
        print("Loading existing model weights.")
        state_dict = torch.load(existing_weights, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    return model


# def get_transforms(input_size: int, preprocess_mode: str, square_pad: bool):
#     """Transformation applied to each image"""

#     if preprocess_mode == "torch":
#         mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#     elif preprocess_mode == "tf":
#         mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
#     else:
#         mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

#     if square_pad:
#         pass


def identity(x):
    """Identity function"""
    return x


# def webdataset_pipeline(
#     sharedurl: str,
#     input_size: int,
#     batch_size: int,
#     preprocess_mode: str,
#     num_workers: int,
#     square_pad: bool,
#     is_training: bool = False,
# ) -> None:
#     """Main dataset builder and loader function"""

#     # Load the webdataset
#     if is_training:
#         dataset = wds.WebDataset(sharedurl, shardshuffle=True)
#         dataset = dataset.shuffle(10000)
#     else:
#         dataset = wds.WebDataset(sharedurl, shardshuffle=False)

#     # Get image transforms
#     img_transform = get_transforms(input_size, preprocess_mode, square_pad)

#     # Decode dataset
#     dataset = (
#         dataset.decode("pil").to_tuple("jpg", "cls").map_tuple(img_transform, identity)
#     )

#     loader = torch.utils.data.DataLoader(
#         dataset, num_workers=num_workers, batch_size=batch_size
#     )

#     pass


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""

    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()
