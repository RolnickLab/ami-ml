"""
Author         : Aditya Jain
Last modified  : May 12th, 2023
About          : List of defined learning rate schedulers
"""

import torch.optim as optim


class LRScheduler:
    def __init__(self, name, optimizer, max_epochs, lr_min=0.0005):
        """
        Args:
            name       : name of the learning rate scheduler
            optimizer  : optimizer being used
            max_epochs : maximum number of epochs
            lr_min     : minimum learning rate
        """
        self.name = name
        self.optimizer = optimizer
        self.T_max = max_epochs
        self.lr_min = lr_min

    def get_scheduler(self):
        if self.name == "cosine_annealing":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.T_max, self.lr_min
            )
        else:
            raise RuntimeError(
                f"{self.name} learning rate scheduler is not implemented."
            )
