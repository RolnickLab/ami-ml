#!/usr/bin/env python
# coding: utf-8

""" Evaluation of models on AMI Traps test set
"""

import os
import pickle
from pathlib import Path
from typing import Literal, Tuple

import dotenv
import torch
from timm.utils import AverageMeter

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.utils import build_model

dotenv.load_dotenv()


def _filter_classes(
    data_dir: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    samples_type: Literal["all", "seen", "unseen"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter classes based on the samples type.

    Args:
        images (torch.Tensor): Tensor of images.
        labels (torch.Tensor): Tensor of labels.
        samples_type (str): Type of samples to filter.

    Returns:
        [torch.Tensor, torch.Tensor]: Filtered labels and corresponding images.
    """
    if samples_type in ["seen", "unseen"]:
        # Load the species list
        with open(Path(data_dir) / f"{samples_type}_species.pkl", "rb") as f:
            species_list = pickle.load(f)

        # Create a mask for filtering
        mask = torch.isin(labels, torch.tensor(species_list).to(labels.device))

        # Filter the images and labels
        labels = labels[mask]
        images = images[mask]

    return images, labels


def evaluate_model(
    data_dir: str,
    model_file: str,
    model_type: str,
    num_classes: int,
    test_webdataset: str,
    samples_type: Literal["all", "seen", "unseen"],
    batch_size: int = 32,
    image_input_size: int = 128,
    preprocess_mode: str = "torch",
) -> None:
    """Evaluate a model on the AMI Traps test set.

    Args:
        model_dir (str): Directory containing the model.
        dataset_dir (str): Directory containing the dataset.
        batch_size (int, optional): Batch size for evaluation. Defaults to 32.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
    """

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device, model_type, num_classes, model_file, checkpoint=False)
    model.eval()

    # Load the test dataset
    test_dataloader = build_webdataset_pipeline(
        test_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
    )

    # Other hyperparameters
    running_accuracy = AverageMeter()

    # Iterate through the test dataset
    with torch.no_grad():
        for batch_data in test_dataloader:
            images, labels = batch_data
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            # Filter out required classes
            images, labels = _filter_classes(data_dir, images, labels, samples_type)

            # Model inference
            outputs = model(images)

            # Calculate and update the accuracy
            _, predicted = torch.max(outputs, 1)
            if labels.size(0) > 0:
                # Update the running accuracy
                running_accuracy.update(
                    (predicted == labels).sum().item() / labels.size(0)
                )

    print(f"Test Accuracy: {running_accuracy.avg:.4f}", flush=True)


if __name__ == "__main__":
    # Get environment variables
    MODEL = os.getenv("BASE_MODEL_UK_DENMARK", "model.pt")
    DATA_DIR = os.getenv("FINE_TUNING_UK_DENMARK_DATA_DIR", "./data")
    TEST_WBDS = os.getenv(
        "FINE_TUNING_UK_DENMARK_TEST_WBDS",
        "./test-000000.tar",
    )

    evaluate_model(DATA_DIR, MODEL, "convnext_tiny_in22k", 2603, TEST_WBDS, "unseen")
