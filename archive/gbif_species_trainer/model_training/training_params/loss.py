"""
Author         : Aditya Jain
Last modified  : May 12th, 2023
About          : List of defined loss functions
"""

import torchvision.models as models
from torch import nn


class Loss:
    def __init__(self, name):
        """
        Args:
            name: the name of the loss function
        """
        self.name = name

    def get_loss(self):
        if self.name == "crossentropy":
            return nn.CrossEntropyLoss()
        else:
            raise RuntimeError(f"{self.name} loss is not implemented.")
