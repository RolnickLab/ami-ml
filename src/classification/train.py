#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

from typing import Optional

import torch

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import (
    build_model,
    get_learning_rate_scheduler,
    get_loss_function,
    get_optimizer,
    set_random_seeds,
)


def _train_model_for_one_epoch() -> None:
    """Training model for one epoch"""


def train_model(
    random_seed: int,
    model_type: str,
    num_classes: int,
    existing_weights: Optional[str],
    total_epochs: int,
    warmup_epochs: int,
    train_webdataset: str,
    val_webdataset: str,
    test_webdataset: str,
    image_input_size: int,
    batch_size: int,
    preprocess_mode: str,
    optimizer_type: str,
    learning_rate: float,
    learning_rate_scheduler_type: str,
    weight_decay: float,
    loss_function_type: str,
    label_smoothing: float,
) -> None:
    """Main training function"""

    # Set random seeds
    set_random_seeds(random_seed)

    # Model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The available device is {device}.")
    model = build_model(device, model_type, num_classes, existing_weights)
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

    # Other training ingredients
    optimizer = get_optimizer(optimizer_type, model, learning_rate, weight_decay)
    steps_per_epoch = ...
    learning_rate_scheduler = get_learning_rate_scheduler(
        optimizer,
        learning_rate_scheduler_type,
        total_epochs,
        steps_per_epoch,
        warmup_epochs,
    )
    loss_function = get_loss_function(
        loss_function_type, label_smoothing=label_smoothing
    )
    print(
        loss_function,
        learning_rate_scheduler,
        optimizer,
        training_dataloader,
        validation_dataloader,
        test_dataloader,
    )

    # Model training
    _train_model_for_one_epoch()
