'''
Author: Aditya Jain
Date  : 19th January, 2022
About : A custom class for imagenet validation dataset
'''
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import json
import torch

class ImagenetValDataset(Dataset):
	def __init__(self, root_dir, label_list, convert_list, transform=None):      
		'''
		Args:
			root_dir (string)  : root directory path that contains the 1000 classes folders
			label_list (string) : path to file containing ground truth labels for each of the val datapoint
			convert_list (string): path to file that contains the mapping from ILSVRC2012_ID to PyTorch IDs
			transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
		self.root_dir     = root_dir
		self.label_list   = pd.read_csv(label_list)
		self.convert_list = pd.read_csv(convert_list)
		self.transform    = transform

	def __len__(self):
		# return size of dataset
		return len(self.label_list)

	def __getitem__(self, idx):
		# returns image and label
		ilsvrc_label = self.label_list.loc[idx, 'validation_imagenet_labels']
		wnid         = self.convert_list.loc[ilsvrc_label-1, 'WNID']
		label        = self.convert_list.loc[ilsvrc_label-1, 'PyTorch_ID']
		num          = str(idx+1)
		image_path   = self.root_dir + wnid + '/' + 'ILSVRC2012_val_000' + num.zfill(5) + '.JPEG'
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