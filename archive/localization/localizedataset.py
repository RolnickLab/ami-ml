"""
Author: Aditya Jain
Date  : 6th August, 2021
About : A custom class for localization dataset
"""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import torch
import os

class LocalizeDataset(Dataset):
	def __init__(self, root_dir, transform=None):      
		'''
		Args:
			root_dir (string)  : root directory path that contains all the data
			transform (callable, optional): Optional transform to be applied
                on a sample.
		'''
		self.root_dir   = root_dir
		self.data_fname = 'dl_train/'               # folder name for the training data
		self.annot_name = 'annotation_data.json'   # annotation name
		self.transform  = transform
		
		self.imgs       = list(sorted(os.listdir(self.root_dir+self.data_fname)))
		self.annot_file = self.root_dir+self.annot_name
		f               = open(self.annot_file)
		self.annot_data = json.load(f) 

	def __len__(self):
		# return size of dataset
		return len(self.imgs)

	def __getitem__(self, idx):
		image_name = self.imgs[idx]
		image_path = self.root_dir + self.data_fname + image_name
		image      = Image.open(image_path)

		bbox_data  = torch.FloatTensor(self.annot_data[image_name][0])
		label_data = torch.LongTensor(self.annot_data[image_name][1])
		image_id   = torch.LongTensor(idx)
		area       = (bbox_data[:, 3] - bbox_data[:, 1]) * (bbox_data[:, 2] - bbox_data[:, 0])
		iscrowd    = torch.zeros((len(label_data),), dtype=torch.int64)
		
		target             = {}
		target["boxes"]    = bbox_data
		target["labels"]   = label_data
		target["image_id"] = image_id
		target["area"]     = area
		target["iscrowd"]  = iscrowd
		
		if self.transform:
			image = self.transform(image)
		
		return image, target