#!/usr/bin/env python
# coding: utf-8

"""
Author       : Aditya Jain
Date Started : November 22, 2022
About        : This is the training file DL-based localization module
"""

import wandb
import torchvision.models as torchmodels
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSD as SSDlite
import torch
from torch.utils.data import random_split
from torch import nn
from torchsummary import summary
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
import random
import argparse

from localizedataset import LocalizeDataset


def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(args):
	"""main function for training"""
	
	# initialise weights and biases api
	wandb.init(project=args.wandb_project, entity=args.wandb_entity)
	wandb.init(settings=wandb.Settings(start_method="fork"))

	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(device)

	# load a model pre-trained pre-trained on COCO
	model = torchvision.models.detection.__dict__[args.model_type](num_classes=2, weights_backbone=args.pretrained)
	model       = model.to(device)

	root_dir    = args.data_path + '/' + 'Localization/'

	batch_size  = args.batch_size
	train_per   = args.train_per/100   
	num_epochs  = args.num_epochs
	early_stop  = args.early_stop
	dtstr       = datetime.datetime.now()
	dtstr       = dtstr.strftime("%Y-%m-%d-%H-%M")
	save_path   = args.save_path + dtstr + '.pt'

	transformer        = transforms.Compose([              
                        transforms.ToTensor()])
	data               = LocalizeDataset(root_dir, transformer)
	train_size         = int(train_per*len(data))
	val_size           = len(data)-train_size

	train_set, val_set = random_split(data, [train_size, val_size])
	train_dataloader   = DataLoader(train_set,batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	val_dataloader     = DataLoader(val_set,batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	model.train()
	lowest_val_loss = 100000000
	early_stp_count = 0

	for epoch in range(num_epochs):
		train_loss = 0
		val_loss   = 0
		s_time     = time.time()
    
		for image_batch, label_batch in train_dataloader:   
			image_batch  = list(image.to(device) for image in image_batch) 
			label_batch  = [{k: v.to(device) for k, v in t.items()} for t in label_batch]

			output       = model(image_batch,label_batch)   
			total_loss   = sum(loss for loss in output.values())
			train_loss   += total_loss.item()
        
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
		print('Len of loader: ', len(train_dataloader))
		train_loss = train_loss/len(train_dataloader)
          
		for image_batch, label_batch in val_dataloader:    
			image_batch  = list(image.to(device) for image in image_batch) 
			label_batch  = [{k: v.to(device) for k, v in t.items()} for t in label_batch]

			output       = model(image_batch,label_batch)   
			total_loss   = sum(loss for loss in output.values())
			val_loss     += total_loss.item() 
		val_loss = val_loss/len(val_dataloader)
		print('Len of loader: ', len(val_dataloader))
        
		if val_loss<lowest_val_loss:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'train_loss': train_loss,
				'val_loss':val_loss}, 
				save_path)                
			lowest_val_loss = val_loss
			early_stp_count = 0
		else:
			early_stp_count += 1 
        
		# logging metrics
		wandb.log({'training loss': train_loss, 'validation loss': val_loss, 'epoch': epoch})
		e_time = (time.time()-s_time)/60   # time taken in minutes 
		wandb.log({'time per epoch': e_time, 'epoch': epoch})

		if early_stp_count >= early_stop:
			break         

def set_random_seed(random_seed):
	"""set random seed for reproducibility"""
	random.seed(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", help = "root directory containing the localization data", required=True)
	parser.add_argument("--model_type", help = "pytorch object detector type", required=True)
	parser.add_argument("--pretrained", help = "pre-training weights", required=True)
	parser.add_argument("--batch_size", help = "batch size for training", default=16, type=int)
	parser.add_argument("--train_per", help = "percentage of data for training", required=True, type=int)
	parser.add_argument("--num_epochs", help = "number of epochs to train for", required=True, type=int)
	parser.add_argument("--early_stop", help = "number of epochs after which to stop training if validation loss does not improve", required=True, type=int)
	parser.add_argument("--save_path", help = "path of the model to save", required=True)
	parser.add_argument("--wandb_project", help = "project name of weights and biases workspace", required=True)
	parser.add_argument("--wandb_entity", help = "entity name of weights and biases workspace", required=True)
	parser.add_argument("--random_seed", help = "random seed for reproducibility", default=42, type=int)
	args   = parser.parse_args()
	
	set_random_seed(args.random_seed)
	train_model(args)



