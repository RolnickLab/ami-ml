"""
Author         : Aditya Jain
Last modified  : May 12th, 2023
About          : List of defined optimizers
"""

from torch import nn
import torch.optim as optim


class Optimizer:
    def __init__(self, name, model, learning_rate, momentum):
        """
        Args:
            name          : name of the optimizer
            model         : model for getting its parameters
            learning_rate : learning rate of the optimizer
            momentum      : momentum of the optimizer
        """
        self.name = name
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum

    def get_optim(self):
        if self.name == "sgd":
            return optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum
            )
        else:
            raise RuntimeError(f"{self.name} optimizer is not implemented.")
