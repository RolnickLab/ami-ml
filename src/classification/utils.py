#!/usr/bin/env python
# coding: utf-8

""" Utility functions
"""

import os
import random

import numpy as np
import timm
import torch

# import webdataset as wds


def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility"""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def model_builder(model_name: str, num_classes: int, pretrained: bool = True):
    """Model builder"""

    if model_name == "timm_efficientnetv2-b3":
        model = timm.create_model(
            "tf_efficientnetv2_b3", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_efficientnetv2-s-in21k":
        model = timm.create_model(
            "tf_efficientnetv2_s_in21k", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_swin-s":
        model = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif model_name == "timm_mobilenetv3large":
        model = timm.create_model(
            "mobilenetv3_large_100", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_resnet50":
        model = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_convnext-t":
        model = timm.create_model(
            "convnext_tiny_in22k", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_convnext-b":
        model = timm.create_model(
            "convnext_base_in22k", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "timm_vit-b16-128":
        model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=pretrained,
            img_size=128,
            num_classes=num_classes,
        )
    elif model_name == "timm_vit-b16-224":
        model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "timm_vit-b16-384":
        model = timm.create_model(
            "vit_base_patch16_384",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        raise RuntimeError(f"Model {model_name} not implemented")

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
