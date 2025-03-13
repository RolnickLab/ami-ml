#!/usr/bin/env python
# coding: utf-8

"""Evaluation of order classifier for binary classification on GBIF order test set
"""


import os

import torch

# 3rd party packages
from dotenv import load_dotenv
from timm.utils import AverageMeter

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import build_model

# Load secrets and config from optional .env file
load_dotenv()


def _update_nonmoth_labels(label: torch.Tensor) -> torch.Tensor:
    """Combine all non-moth taxas to just one label"""

    label[label != 0] = 1

    return label


def _evaluate_model(
    model: torch.nn.Module,
    device: str,
    loss_function: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    set_type: str,
) -> dict:
    """Evaluate model either for validation or test set"""

    running_loss = AverageMeter()
    running_accuracy = AverageMeter()

    model.eval()
    for batch_data in dataloader:
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
        labels = _update_nonmoth_labels(labels)
        predicted = _update_nonmoth_labels(predicted)
        running_accuracy.update((predicted == labels).sum().item() / labels.size(0))

    metrics = {
        f"{set_type}_loss": round(running_loss.avg, 4),
        f"{set_type}_accuracy": round(running_accuracy.avg * 100, 2),
    }

    return metrics


def binary_model_evaluation(
    run_name: str,
    model_weights: str,
    model_type: str,
    num_classes: int,
    test_webdataset: str,
    image_input_size: int = 128,
):
    """Main function for binary model evaluation on GBIF Test set"""

    print(f"Model evaluation for {run_name}.")

    # Model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The available device is {device}.")
    model = build_model(device, model_type, num_classes, model_weights, checkpoint=True)

    # Load dataloader
    test_dataloader = build_webdataset_pipeline(
        test_webdataset,
        image_input_size,
        batch_size=32,
        preprocess_mode="torch",
    )

    # Evaluate the model on test data
    loss_function = torch.nn.CrossEntropyLoss()
    eval_metrics = _evaluate_model(
        model, device, loss_function, test_dataloader, "test"
    )

    print(f"Test metrics are: {eval_metrics}")


if __name__ == "__main__":
    ORDER_MODEL = os.environ.get("ORDER_MODEL")
    ORDER_RUN_NAME = os.environ.get("ORDER_RUN_NAME")
    ORDER_TEST_WBDS_EVAL = os.environ.get("ORDER_TEST_WBDS_EVAL")

    binary_model_evaluation(
        run_name=ORDER_RUN_NAME,
        model_weights=ORDER_MODEL,
        model_type="convnext_tiny_in22k",
        num_classes=16,
        test_webdataset=ORDER_TEST_WBDS_EVAL,
    )
