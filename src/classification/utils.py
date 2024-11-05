#!/usr/bin/env python
# coding: utf-8

""" Utility functions
"""

import random
import typing as tp

import numpy as np
import timm
import torch

from src.classification.constants import AVAILABLE_MODELS, VIT_B16_128


def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility"""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


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


def build_model(
    device: str,
    model_type: str,
    num_classes: int,
    existing_weights: tp.Optional[str],
    pretrained: bool = True,
):
    """Model builder"""

    if model_type not in AVAILABLE_MODELS:
        raise RuntimeError(f"Model {model_type} not implemented")

    model_arguments = {"pretrained": pretrained, "num_classes": num_classes}
    if model_type == VIT_B16_128:
        # There is no off-the-shelf ViT model for 128x128 image size,
        # so we use 224x224 model with a custom input image size
        model_type = "vit_base_patch16_224_in21k"
        model_arguments["img_size"] = 128

    model = timm.create_model(model_type, **model_arguments)

    # If available, load existing weights
    if existing_weights:
        print("Loading existing model weights.")
        state_dict = torch.load(existing_weights, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)

    # Make use of multiple GPUs, if available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    return model
