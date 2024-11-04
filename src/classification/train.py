#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

from typing import Optional

import torch

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.models import model_builder
from src.classification.utils import set_random_seeds


def _train_model_for_one_epoch():
    """Training model for one epoch"""


def train_model(
    random_seed: int,
    model_type: str,
    num_classes: int,
    existing_weights: Optional[str],
    train_webdataset: str,
    val_webdataset: str,
    test_webdataset: str,
    image_input_size: int,
    batch_size: int,
    preprocess_mode: str,
) -> None:
    """Main training function"""

    # Set random seeds
    set_random_seeds(random_seed)

    # Model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The available device is {device}.")
    model = model_builder(device, model_type, num_classes, existing_weights)
    print(model)

    # Setup dataloaders
    training_dataloader = build_webdataset_pipeline(
        train_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
        is_training=True,
    )
    validation_dataloader = build_webdataset_pipeline(
        val_webdataset, image_input_size, batch_size, preprocess_mode
    )
    test_dataloader = build_webdataset_pipeline(
        test_webdataset, image_input_size, batch_size, preprocess_mode
    )
    print(training_dataloader, validation_dataloader, test_dataloader)
