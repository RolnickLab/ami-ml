#!/usr/bin/env python
# coding: utf-8

# In[20]:


"""
Author       : Aditya Jain
Date Started : May 11, 2022
About        : This file does DL-based localization on raw images and saves annotation information
"""

import torch
import torchvision.models as torchmodels
import torchvision
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import cv2
import json


# #### User-defined variables

# In[ ]:


data_path  = '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/2022_05_14/'
save_path  = '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/'
annot_file = 'localiz_annotation-2022_05_14.json'


# #### Model Loading

# In[21]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# load a model pre-trained pre-trained on COCO
model       = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


model_path  = '/home/mila/a/aditya.jain/logs/v1_localizmodel_2021-08-17-12-06.pt'
checkpoint  = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


# In[ ]:


model       = model.to(device)
model.eval()

annot_data = {}
SCORE_THR  = 0.99
image_list = os.listdir(data_path)
image_list.sort()

transform  = transforms.Compose([              
            transforms.ToTensor()])

for img in image_list:
    image_path = data_path + img
    image      = transform(Image.open(image_path))
    image_pred = torch.unsqueeze(image, 0).to(device)
    output     = model(image_pred)
    
    bboxes     = output[0]['boxes'][output[0]['scores'] > SCORE_THR]    
    bbox_list  = []
    label_list = []
    
    for box in bboxes:
        box_numpy = box.detach().cpu().numpy() 
        bbox_list.append([int(box_numpy[0]), int(box_numpy[1]),                           int(box_numpy[2]), int(box_numpy[3])])
        label_list.append(1)
        
    annot_data[img] = [bbox_list, label_list]

with open(save_path + annot_file , 'w') as outfile:
    json.dump(annot_data, outfile)    


# In[ ]:




