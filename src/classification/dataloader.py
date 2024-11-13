#!/usr/bin/env python
# coding: utf-8

""" Functions related to dataset loading and image transformations
"""

import os
from functools import partial
from typing import Any

import numpy as np
import PIL
import torch
import webdataset as wds
from torchvision import transforms


def _pad_to_square(image: PIL.Image.Image) -> PIL.Image.Image:
    """Padding transformation to make the image square"""

    width, height = image.size
    if height < width:
        transform = transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        transform = transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        transform = transforms.Pad(padding=[0, 0, 0, 0])

    return transform(image)


def _normalization(preprocess_mode: str) -> tuple[list[float], list[float]]:
    """Get the mean and std for normalization"""

    if preprocess_mode == "torch":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif preprocess_mode == "tf":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    return mean, std


def _random_resize(image: PIL.Image.Image, full_size: int) -> PIL.Image.Image:
    """Mixed resolution transformation"""

    random_num = np.random.uniform()
    if random_num <= 0.25:
        transform = transforms.Resize((int(0.5 * full_size), int(0.5 * full_size)))
        image = transform(image)
    elif random_num <= 0.5:
        transform = transforms.Resize((int(0.25 * full_size), int(0.25 * full_size)))
        image = transform(image)

    return image


def _get_transforms(
    input_size: int, is_training: bool, preprocess_mode: str = "torch"
) -> transforms.Compose:
    """Transformation applied to each image"""

    # Add square padding
    final_transforms = [transforms.Lambda(_pad_to_square)]

    if is_training:
        f_random_resize = partial(_random_resize, full_size=input_size)
        final_transforms += [
            transforms.Lambda(f_random_resize),  # mixed resolution
            transforms.RandomResizedCrop(input_size, scale=(0.3, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]
    else:
        final_transforms += [transforms.Resize((input_size, input_size))]

    # Normalization
    mean, std = _normalization(preprocess_mode)
    final_transforms += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(final_transforms)


def build_webdataset_pipeline(
    sharedurl: str,
    input_size: int,
    batch_size: int,
    preprocess_mode: str,
    is_training: bool = False,
) -> torch.utils.data.DataLoader:
    """Main dataset builder and loader function"""

    # Load the webdataset
    if is_training:
        dataset = wds.WebDataset(sharedurl, shardshuffle=True)
        dataset = dataset.shuffle(10000)
    else:
        dataset = wds.WebDataset(sharedurl, shardshuffle=False)

    # Get image transforms
    image_transform = _get_transforms(input_size, is_training, preprocess_mode)

    # Decode dataset
    dataset_decoded = (
        dataset.decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(image_transform, _identity)
    )

    # Create dataLoader
    dataset_loader = torch.utils.data.DataLoader(
        dataset_decoded, num_workers=_get_num_workers(), batch_size=batch_size
    )

    return dataset_loader


def _identity(x: Any) -> Any:
    """Identity function"""
    return x


def _get_num_workers() -> int:
    """Gets the optimal number of dataloader workers to use in the current job"""

    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()
