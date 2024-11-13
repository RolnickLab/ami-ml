#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

import typing as tp
from typing import Optional

import torch

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import (
    build_model,
    get_learning_rate_scheduler,
    get_loss_function,
    get_optimizer,
    get_webdataset_length,
    set_random_seeds,
)

total_train_steps = 0


def _save_model_checkpoint(model: torch.nn.Module, model_path: str) -> None:
    """Save model to disk"""
    pass


def _train_model_for_one_epoch(
    model: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    learning_rate_scheduler: Optional[tp.Any],
) -> None:
    """Training model for one epoch"""
    global total_train_steps

    model.train()
    for batch_data in train_dataloader:
        images, labels = batch_data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # Learning rate scheduler step
        if learning_rate_scheduler:
            total_train_steps += 1
            learning_rate_scheduler.step_update(num_updates=total_train_steps)

        # TODO: Calculate accuracy metrics
        # TODO: Take loss average before returning

    return loss


def _validate_model(
    model: torch.nn.Module,
    device: str,
    loss_function: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
) -> None:
    """Validate model after one epoch"""

    model.eval()
    for batch_data in val_dataloader:
        images, labels = batch_data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        with torch.no_grad():
            outputs = model(images)
            loss = loss_function(outputs, labels)

        # TODO: Take loss average before returning

    return loss


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
    learning_rate_scheduler_type: Optional[str],
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

    # Setup dataloaders
    train_dataloader = build_webdataset_pipeline(
        train_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
        is_training=True,
    )
    val_dataloader = build_webdataset_pipeline(
        val_webdataset, image_input_size, batch_size, preprocess_mode
    )
    # test_dataloader = build_webdataset_pipeline(
    #     test_webdataset, image_input_size, batch_size, preprocess_mode
    # )

    # Other training ingredients
    optimizer = get_optimizer(optimizer_type, model, learning_rate, weight_decay)
    learning_rate_scheduler = None
    if learning_rate_scheduler_type:
        train_data_length = get_webdataset_length(train_webdataset)
        steps_per_epoch = int((train_data_length - 1) / batch_size) + 1
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

    # Model training
    for epoch in range(1, total_epochs + 1):
        _train_model_for_one_epoch(
            model,
            device,
            optimizer,
            loss_function,
            train_dataloader,
            learning_rate_scheduler,
        )
        _validate_model(model, device, loss_function, val_dataloader)

        # TODO: Save model checkpoint
        # TODO: Calculate accuracy metrics
        # TODO: Receive epoch-level metrics and upload to W&B


if __name__ == "__main__":
    train_model(
        random_seed=42,
        model_type="resnet50",
        num_classes=29176,
        existing_weights=None,
        total_epochs=10,
        warmup_epochs=1,
        train_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/train/train450-{000000..000001}.tar",
        val_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/val/val450-000000.tar",
        test_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/test/test450-000000.tar",
        image_input_size=128,
        batch_size=16,
        preprocess_mode="torch",
        optimizer_type="adamw",
        learning_rate=0.001,
        learning_rate_scheduler_type="cosine",
        weight_decay=1e-5,
        loss_function_type="cross_entropy",
        label_smoothing=0.1,
    )
