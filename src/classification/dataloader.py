#!/usr/bin/env python
# coding: utf-8

""" Functions related to dataset loading and image transformations
"""

import os
import random
from functools import partial
from typing import Any

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

    preprocess_params = {
        "torch": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "tf": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "default": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    }

    mean, std = preprocess_params.get(preprocess_mode, preprocess_params["default"])

    return mean, std


def _mixed_resolution(image: PIL.Image.Image, full_size: int) -> PIL.Image.Image:
    """Mixed resolution data augmentation technique"""

    values = [0.25, 0.5, None, None]
    random_value = random.choice(values)

    if random_value:
        transform = transforms.Resize(
            (int(random_value * full_size), int(random_value * full_size))
        )
        image = transform(image)

    return image


def _get_transforms(
    input_size: int,
    is_training: bool,
    mixed_resolution_data_aug: bool,
    preprocess_mode: str = "torch",
) -> transforms.Compose:
    """Transformation applied to each image"""

    # Add square padding
    final_transforms = [transforms.Lambda(_pad_to_square)]

    # Mixed resolution data augmentation. This randomly downscales the image to simulate small,
    # low-resolution subjects (a camera-trap degradation). It defaults on in the CLI, but the
    # Fieldguide corpus is DSLR/phone photos cropped tight to the subject, so this degradation does
    # not match the serving domain. MIXED_RESOLUTION_AUG in the environment overrides the caller:
    # set it to 0 to force the augmentation off (the CLI flag cannot, being a click bool that reads
    # any non-empty string as True).
    _mixres_env = os.environ.get("MIXED_RESOLUTION_AUG")
    if _mixres_env is not None:
        mixed_resolution_data_aug = _mixres_env == "1"
    if mixed_resolution_data_aug:
        f_random_resize = partial(_mixed_resolution, full_size=input_size)
        final_transforms += [transforms.Lambda(f_random_resize)]

    # Standard training augmentations
    if is_training:
        # RandomResizedCrop scale floor: the default 0.3 zooms into a 30%-area sub-crop, which on
        # the Fieldguide corpus (already cropped tight to the subject) can cut off wings/antennae
        # and does not match the serving path (a whole-image resize, no random crop). RRC_SCALE_MIN
        # raises the floor to keep crops near the full subject.
        _rrc_min = float(os.environ.get('RRC_SCALE_MIN', '0.3'))
        final_transforms += [
            transforms.RandomResizedCrop(input_size, scale=(_rrc_min, 1)),
            transforms.RandomHorizontalFlip(),
        ]
        # RandAugment is the dominant per-image CPU cost; RANDAUGMENT=0 drops it for a faster,
        # less-regularised run. On by default.
        if os.environ.get('RANDAUGMENT', '1') == '1':
            final_transforms += [transforms.RandAugment(num_ops=2, magnitude=9)]
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
    mixed_resolution_data_aug: bool = False,
    is_training: bool = False,
) -> torch.utils.data.DataLoader:
    """Main dataset builder and loader function"""

    # Load the webdataset
    dataset = wds.WebDataset(sharedurl, shardshuffle=is_training, empty_check=False)
    if is_training:
        dataset = dataset.shuffle(10000)

    # Get image transforms
    image_transform = _get_transforms(
        input_size, is_training, mixed_resolution_data_aug, preprocess_mode
    )

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
