'''
Author: Aditya Jain
Date  : 14th May, 2021
About : Various loss functions defined
'''

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