"""
Author              : Aditya Jain
Date last modified  : August 24, 2023
About               : This script predicts the class given an input image
"""

import pandas as pd
import os
import shutil

# Import modules
import sys, os
sys.path.append(
    "/home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/models"
)

import PIL
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from resnet50 import Resnet50
import json
import numpy as np

def padding(image: PIL.Image.Image):
    """returns the padding transformation required based on image shape"""

    height, width = np.shape(image)[0], np.shape(image)[1]

    if height < width:
        pad_transform = transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        pad_transform = transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        return None

    return pad_transform

# User-defined variables
image_path = "/home/mila/a/aditya.jain/mothAI/gbif_species_trainer/utils/moth-uoft.jpeg"
config_file = "/home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/config/01-config_quebec-vermont.json"
model_path = "/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v07_resnet50_2022-12-22-07-54.pt"
label_file = "/home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/quebec-vermont_numeric_labels.json"
img_resize = 300

# Load configuration data
f = open(config_file)
config_data = json.load(f)
num_classes = config_data["model"]["species_num_classes"]

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Resnet50(num_classes).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Image loading and processing
image = Image.open(image_path)
transform = transforms.Compose([transforms.ToTensor()])
image = transform(image)
image = torch.unsqueeze(image, 0).to(device)
# padding_transform = padding(image)
# if padding_transform:
#     image = padding_transform(image)
resize_transform = transforms.Compose([transforms.Resize((img_resize, img_resize))])
image = resize_transform(image)

# Model prediction
prediction = model(image)
conf, predict_indx = torch.topk(prediction, 3)

# Get the species name
f            = open(label_file)
label_info   = json.load(f)
species_list = label_info["species_list"]

print('The predicted species is : ', species_list[predict_indx])
