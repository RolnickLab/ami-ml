#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

from typing import Optional

import torch

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import build_model, get_optimizer, set_random_seeds


def _train_model_for_one_epoch() -> None:
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
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
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
    # learning_rate_scheduler = get_learning_rate_scheduler()
    # loss = ...
    print(optimizer, training_dataloader, validation_dataloader, test_dataloader)

    # Model training
    _train_model_for_one_epoch()


if __name__ == "__main__":
    train_model(
        random_seed=42,
        model_type="vit_base_patch16_128_in21k",
        num_classes=29176,
        existing_weights=None,
        train_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/train/train450-000000.tar",
        val_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/val/val450-000000.tar",
        test_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/test/test450-000000.tar",
        image_input_size=128,
        batch_size=1,
        preprocess_mode="torch",
        optimizer_type="adamw",
        learning_rate=0.001,
        weight_decay=1e-5,
    )
