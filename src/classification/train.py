#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

import pathlib
import time
import typing as tp
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from timm.utils import AverageMeter

import wandb
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
) -> tuple[dict, int]:  # TODO: First element will eventually turn into a dict
    """Training model for one epoch"""

    total_train_steps_current = total_train_steps
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()

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

        # Calculate the batch accuracy and update to global accuracy
        _, predicted = torch.max(outputs, 1)
        running_accuracy.update((predicted == labels).sum().item() / labels.size(0))

        # Learning rate scheduler step
        if learning_rate_scheduler:
            total_train_steps_current += 1
            learning_rate_scheduler.step_update(num_updates=total_train_steps_current)

    metrics = {"train_loss": running_loss.avg, "train_accuracy": running_accuracy.avg}

    return metrics, total_train_steps_current


def _evaluate_model(
    model: torch.nn.Module,
    device: str,
    loss_function: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    set_type: str,
) -> dict:
    """Evaluate model either for validation or test set"""

    running_loss = AverageMeter()
    running_accuracy = AverageMeter()

    model.eval()
    for batch_data in val_dataloader:
        images, labels = batch_data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        with torch.no_grad():
            outputs = model(images)
            loss = loss_function(outputs, labels)

        # Calculate the average loss per sample
        running_loss.update(loss.item())

        # Calculate the batch accuracy and update to global accuracy
        _, predicted = torch.max(outputs, 1)
        running_accuracy.update((predicted == labels).sum().item() / labels.size(0))

    metrics = {
        f"{set_type}_loss": running_loss.avg,
        f"{set_type}_accuracy": running_accuracy.avg,
    }

    return metrics


def train_model(
    random_seed: int,
    model_type: str,
    num_classes: int,
    existing_weights: Optional[str],
    total_epochs: int,
    warmup_epochs: int,
    early_stopping: int,
    train_webdataset: str,
    val_webdataset: str,
    test_webdataset: str,
    image_input_size: int,
    batch_size: int,
    preprocess_mode: str,
    optimizer_type: str,
    learning_rate: float,
    learning_rate_scheduler: Optional[str],
    weight_decay: float,
    loss_function_type: str,
    weight_on_order_loss: float,
    label_smoothing: float,
    mixed_resolution_data_aug: bool,
    model_save_directory: str,
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
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
        mixed_resolution_data_aug=mixed_resolution_data_aug,
        is_training=True,
    )
    val_dataloader = build_webdataset_pipeline(
        val_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
    )
    test_dataloader = build_webdataset_pipeline(
        test_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
    )

    # Other training ingredients
    optimizer = get_optimizer(optimizer_type, model, learning_rate, weight_decay)
    learning_rate_scheduler = None
    if learning_rate_scheduler:
        train_data_length = get_webdataset_length(train_webdataset)
        steps_per_epoch = int((train_data_length - 1) / batch_size) + 1
        learning_rate_scheduler = get_learning_rate_scheduler(
            optimizer,
            learning_rate_scheduler,
            total_epochs,
            steps_per_epoch,
            warmup_epochs,
        )
    loss_function = get_loss_function(
        loss_function_type,
        label_smoothing=label_smoothing,
        weight_on_order=weight_on_order_loss,
    )
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = Path(model_save_directory) / f"{model_type}_{current_date}"

    # Start W&B logging
    if wandb_entity or wandb_project:
        training_configuration = {
            "random_seed": random_seed,
            "model_type": model_type,
            "num_classes": num_classes,
            "existing_weights": existing_weights,
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "early_stopping": early_stopping,
            "train_webdataset": train_webdataset,
            "val_webdataset": val_webdataset,
            "test_webdataset": test_webdataset,
            "image_input_size": image_input_size,
            "batch_size": batch_size,
            "preprocess_mode": preprocess_mode,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": learning_rate_scheduler,
            "weight_decay": weight_decay,
            "loss_function_type": loss_function_type,
            "label_smoothing": label_smoothing,
            "model_save_directory": model_save_directory,
        }
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            config=training_configuration,
        )

    # Model training
    total_train_steps = 0  # total training batches processed
    early_stopping_count = 0
    lowest_val_loss = 1e8
    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        train_metrics, total_train_steps_current = _train_model_for_one_epoch(
            model,
            device,
            optimizer,
            loss_function,
            train_dataloader,
            learning_rate_scheduler,
            total_train_steps,
        )
        total_train_steps = total_train_steps_current
        early_stopping_count += 1
        val_metrics = _evaluate_model(
            model, device, loss_function, val_dataloader, "val"
        )

        if val_metrics["val_loss"] < lowest_val_loss:
            _save_model_checkpoint(
                model,
                model_save_path,
                optimizer,
                learning_rate_scheduler,
                epoch,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
            )
            lowest_val_loss = val_metrics["val_loss"]
            early_stopping_count = 0

        print(
            f"Epoch [{epoch:02d}/{total_epochs}]: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Train Accuracy: {train_metrics['train_accuracy']*100:.2f}%, "
            f"Val Accuracy: {val_metrics['val_accuracy']*100:.2f}%, "
            f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        if wandb_entity or wandb_project:
            wandb.log(
                {
                    "epoch": epoch,
                    "time_per_epoch_mins": (time.time() - epoch_start_time) / 60,
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "train_accuracy": train_metrics["train_accuracy"],
                    "val_accuracy": val_metrics["val_accuracy"],
                }
            )

        if early_stopping_count >= early_stopping:
            print(
                f"Early stopping at epoch {epoch} with lowest validation loss: {lowest_val_loss:.4f}.",
                flush=True,
            )
            break

    # Evaluate the model on test data
    test_metrics = _evaluate_model(
        model, device, loss_function, test_dataloader, "test"
    )
    print(f"The test accuracy is {test_metrics['test_accuracy']*100:.2f}%.", flush=True)

    if wandb_entity or wandb_project:
        wandb.log({"test_accuracy": test_metrics["test_accuracy"]})
        wandb.log_artifact(
            f"{model_save_path}_checkpoint.pt", type="model", name=wandb_run_name
        )
        wandb.finish()
