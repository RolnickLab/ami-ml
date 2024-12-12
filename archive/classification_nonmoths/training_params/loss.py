"""
Author: Aditya Jain
Date  : May 3, 2022
About : Loss functions defined
"""

import torchvision.models as models
from torch import nn

class Loss():
	def __init__(self, name):
		"""
        Args:
            name: the name of the loss function
        """
		self.name = name

	def func(self):
		if self.name == 'crossentropy':
			return nn.CrossEntropyLoss()