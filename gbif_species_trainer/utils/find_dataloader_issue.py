#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""
Author        : Aditya Jain
Date Started  : June 22, 2022
About         : Helps find data loading issue for some images
"""
import os
import glob
from PIL import Image
from torchvision import transforms, utils
import torchvision.models as torchmodels
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchsummary import summary

from mothdataset import MOTHDataset

root_dir     = '/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk/'
train_set    = '/home/mila/a/aditya.jain/mothAI/classification_moths/data/01-uk-train-split.csv'
label_list   = '/home/mila/a/aditya.jain/mothAI/classification_moths/data/uk_numeric_labels.json'
image_resize = 224
batch_size   = 2


# In[10]:


train_transformer = transforms.Compose([
                        transforms.Resize((image_resize, image_resize)),              # resize the image to 300x300 
                        transforms.ToTensor()])
#                         transforms.RandomHorizontalFlip()])
train_data        = MOTHDataset(root_dir, train_set, label_list, train_transformer)
train_dataloader  = DataLoader(train_data,batch_size=batch_size, shuffle=True)


# In[11]:


for image_batch, label_batch in train_dataloader:    
    continue

