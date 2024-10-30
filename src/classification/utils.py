#!/usr/bin/env python
# coding: utf-8

""" Utility functions
"""

import random
import typing as tp

import numpy as np
import torch


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
