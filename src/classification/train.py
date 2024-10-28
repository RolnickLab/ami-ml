#!/usr/bin/env python
# coding: utf-8


""" Main script for training classification models
"""

# package imports
import torch

# 3rd party packages
from dotenv import load_dotenv

from src.classification.utils import set_random_seeds

# Load secrets and config from optional .env file
load_dotenv()


def train_model_one_epoch():
    """Training model for one epoch"""


def prepare_dataloader():
    """Returns the training, validation and test data loaders,
     which have different transforms
    (data augmentation is only applied on the training set)
    """


def train_model(random_seed: int) -> None:
    """Main training function"""

    # Basic initialization
    set_random_seeds(random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The available device is {device}.")
