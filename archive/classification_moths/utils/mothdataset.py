"""
Author: Aditya Jain
Date  : 7th May, 2021
About : A custom class for moth dataset
"""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import json
import torch

class MOTHDataset(Dataset):
	def __init__(self, root_dir, data_list, label_list, transform=None):      
		"""
		Args:
			root_dir (string)  : root directory path that contains all the data
			data_list (string) : Contains the list of data points for a particular set (train/val/test)
			label_list (string): path to file that contains the list of labels for conversion to numerics
			transform (callable, optional): Optional transform to be applied
                on a sample.
        """
		self.root_dir   = root_dir
		self.data_list  = pd.read_csv(data_list)
		self.transform  = transform
		f               = open(label_list)
		self.label_list = json.load(f) 

	def __len__(self):
		# return size of dataset
		return len(self.data_list)
	
	def __getitem__(self, idx):
		# returns image and label	
		image_data = self.data_list.iloc[idx, :]
		image_path = self.root_dir + image_data['family'] + '/' + image_data['genus'] + '/' + image_data['species'] + '/' + image_data['filename']
		image      = Image.open(image_path)
		if self.transform:
			try:
				print(image_path)
				image  = self.transform(image)
			except:
				print('This image is causing issue: ', image_path)
		return image