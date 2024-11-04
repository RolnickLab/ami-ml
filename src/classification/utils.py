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


def get_optimizer(
    optimizer_type: str,
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Optimizer definitions"""

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise RuntimeError(f"{optimizer_type} optimizer is not implemented.")


def get_learning_rate_scheduler() -> None:
    """Scheduler definitions"""
    pass
