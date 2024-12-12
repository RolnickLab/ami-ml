'''
Author: Aditya Jain
Date  : 28th January, 2022
About : A custom class for loading hard examples
'''
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import json
import torch

class HardExDataset(Dataset):
	def __init__(self, root_dir, label_list, transform=None):      
		'''
		Args:
			root_dir (string)  : root directory containing the hard examples class folders
			label_list (string) : path to file containing ground truth labels for each of the hard ex datapoint
			transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
		self.root_dir     = root_dir
		self.label_list   = pd.read_csv(label_list)
		self.transform    = transform

	def __len__(self):
		# return size of dataset
		return len(self.label_list)
	
	def __getitem__(self, idx):
		# returns image and label
		category     = self.label_list.loc[idx, 'Name_ID']
		file         = self.label_list.loc[idx, 'Filename']
		label        = self.label_list.loc[idx, 'PyTorch_ID']
		
		image_path   = self.root_dir + category + '/' + file
		image        = Image.open(image_path)
		image_shape  = np.shape(image)
		
		if len(image_shape)==2:           # it is a grayscale image; third channel not there
			to_rgb    = transforms.Grayscale(num_output_channels=3)
			image     = to_rgb(image)

		if len(image_shape)>2:
			if image_shape[2]>3:  # RGBA image; extra alpha channel
				image  = image.convert('RGB')

		if self.transform:
			image  = self.transform(image)

		label     = torch.LongTensor([label])

		return image, label