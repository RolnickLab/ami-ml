#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

import pathlib
import typing as tp
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from timm.utils import AverageMeter

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import (
    build_model,
    get_learning_rate_scheduler,
    get_loss_function,
    get_optimizer,
    get_webdataset_length,
    set_random_seeds,
)


def _save_model_checkpoint(
    model: torch.nn.Module,
    model_save_path: pathlib.Path,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: tp.Any,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """Save model to disk"""

    if torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    model_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": learning_rate_scheduler.state_dict()
        if learning_rate_scheduler is not None
        else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(model_checkpoint, f"{model_save_path}_checkpoint.pt")


def _train_model_for_one_epoch(
    model: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    learning_rate_scheduler: Optional[tp.Any],
    total_train_steps: int,
) -> tuple[float, int]:  # TODO: First element will eventually turn into a dict
    """Training model for one epoch"""

    total_train_steps_current = total_train_steps
    running_loss = AverageMeter()

    model.train()
    for batch_data in train_dataloader:
        images, labels = batch_data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        # Forward pass, loss calculation, backward pass, and optimizer step
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate the average loss per sample
        running_loss.update(loss.item())

        # Learning rate scheduler step
        if learning_rate_scheduler:
            total_train_steps_current += 1
            learning_rate_scheduler.step_update(num_updates=total_train_steps_current)

        # TODO: Calculate accuracy metrics
        # TODO: Take loss average before returning

    return running_loss.avg, total_train_steps_current


def _validate_model(
    model: torch.nn.Module,
    device: str,
    loss_function: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
) -> float:
    """Validate model after one epoch"""

    running_loss = AverageMeter()

    model.eval()
    for batch_data in val_dataloader:
        images, labels = batch_data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        with torch.no_grad():
            outputs = model(images)
            loss = loss_function(outputs, labels)

        running_loss.update(loss.item())

    return running_loss.avg


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
    model_save_directory: str,
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
    current_date = datetime.now().date().strftime("%Y%m%d")
    model_save_path = Path(model_save_directory) / f"{model_type}_{current_date}"

    # Model training
    total_train_steps = 0  # total training batches processed
    lowest_val_loss = 1e8
    for epoch in range(1, total_epochs + 1):
        current_train_loss, total_train_steps_current = _train_model_for_one_epoch(
            model,
            device,
            optimizer,
            loss_function,
            train_dataloader,
            learning_rate_scheduler,
            total_train_steps,
        )
        total_train_steps = total_train_steps_current
        current_val_loss = _validate_model(model, device, loss_function, val_dataloader)

        if current_val_loss < lowest_val_loss:
            _save_model_checkpoint(
                model,
                model_save_path,
                optimizer,
                learning_rate_scheduler,
                epoch,
                current_train_loss,
                current_val_loss,
            )
            lowest_val_loss = current_val_loss

        # TODO: Calculate accuracy metrics
        # TODO: Receive epoch-level metrics and upload to W&B


if __name__ == "__main__":
    train_model(
        random_seed=42,
        model_type="resnet50",
        num_classes=29176,
        existing_weights=None,
        total_epochs=1,
        warmup_epochs=0,
        train_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/train/train450-{000000..000001}.tar",
        val_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/val/val450-000000.tar",
        test_webdataset="/home/mila/a/aditya.jain/scratch/global_model/webdataset/test/test450-000000.tar",
        image_input_size=128,
        batch_size=64,
        preprocess_mode="torch",
        optimizer_type="adamw",
        learning_rate=0.001,
        learning_rate_scheduler_type="cosine",
        weight_decay=1e-5,
        loss_function_type="cross_entropy",
        label_smoothing=0.1,
        model_save_directory="/home/mila/a/aditya.jain/scratch",
    )
