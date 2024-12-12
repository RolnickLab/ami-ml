"""
Author: Aditya Jain
Date  : May 3, 2022
About : A custom class for nonmoth classifier
"""
import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import json
import torch

class NonMothDataset(Dataset):
	def __init__(self, root_dir, data_list, transform=None):      
		"""
		Args:
			root_dir (string)  : root directory path that contains all the data
			data_list (string) : Contains the list of data points for a particular set (train/val/test)
			transform (callable, optional): Optional transform to be applied
                on a sample.
        """
		self.root_dir   = root_dir
		self.data_list  = pd.read_csv(data_list)
		self.transform  = transform

	def __len__(self):
		# return size of dataset
		return len(self.data_list)
	
	def __getitem__(self, idx):
		# returns image and label	
		image_data = self.data_list.iloc[idx, :]
		class_name = image_data['class']

		if class_name=='moth':
			image_path = self.root_dir + 'moths/' + image_data['family'] + '/' + image_data['genus'] + '/' + image_data['species'] + '/' + image_data['filename']
			label      = 0
		else:
			image_path = self.root_dir + 'nonmoths/' + image_data['family'] + '/' + image_data['filename']
			label      = 1

		image      = Image.open(image_path)
		if self.transform:
			image  = self.transform(image)

		label        = torch.LongTensor([label])

		if image.shape[0]>3:  # RGBA image; extra alpha channel
			image  = image[0:3,:,:]

		if image.shape[0]==1: # grayscale image; converted to 3 channels r=g=b
			to_pil    = transforms.ToPILImage()
			to_rgb    = transforms.Grayscale(num_output_channels=3)
			to_tensor = transforms.ToTensor()
			image     = to_tensor(to_rgb(to_pil(image)))
			
		return image, label