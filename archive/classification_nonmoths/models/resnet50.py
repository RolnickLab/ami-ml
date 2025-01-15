"""
Author: Aditya Jain
Date  : May 3, 2022
About : Description of ResNet-50 Model
"""

import torchvision.models as models
from torch import nn

class Resnet50(nn.Module):
	def __init__(self, config):
		"""
        Args:
            config: provides parameters for model generation
        """
		super(Resnet50, self).__init__()
		self.num_classes = config['model']['num_classes']
		self.backbone    = models.resnet50(pretrained=True)
		out_dim          = self.backbone.fc.in_features

		self.backbone    = nn.Sequential(*list(self.backbone.children())[:-2])
		self.avgpool     = nn.AdaptiveAvgPool2d(output_size=(1,1))
		self.classifier  = nn.Linear(out_dim, self.num_classes, bias=False)
        
	def forward(self, x):
		x = self.backbone(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		
		return x
	
	def print_test(self):
		return self.val
        