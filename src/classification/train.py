#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

from typing import Optional

import torch

# from src.classification.dataloader import webdataset_pipeline
from src.classification.models import model_builder
from src.classification.utils import set_random_seeds


def prepare_dataloader():
    """Returns the training, validation and test data loaders,
     which have different transforms
    (data augmentation is only applied on the training set)
    """


def train_model_one_epoch():
    """Training model for one epoch"""


def train_model(
    random_seed: int, model_type: str, num_classes: int, existing_weights: Optional[str]
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
    # train_data = webdataset_pipeline()
