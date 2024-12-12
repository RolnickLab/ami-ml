#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
Author        : Aditya Jain
Date Started  : June 24, 2022
About         : Finds and deletes images that are not resizable i.e. corrupted
"""
import os
import glob
from PIL import Image
import pandas as pd
from torchvision import transforms, utils

data_dir    = '/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk/'    
image_resize = 300


file_path      = '/home/mila/a/aditya.jain/mothAI/classification_moths/data/01-uk-test-split.csv'
data           = pd.read_csv(file_path, index_col=False)

tot_img_issues = 0
transformer    = transforms.Compose([
                transforms.Resize((image_resize, image_resize)),              # resize the image to 300x300 
                transforms.ToTensor()])

for indx in data.index:
    filename = data_dir + data['family'][indx] + '/' + data['genus'][indx] + \
    '/' + data['species'][indx] + '/' + data['filename'][indx]
    
    try:
        image = Image.open(filename) 
        image = transformer(image)
    except:
        print(filename)
        tot_img_issues += 1
        os.remove(filename)

print(f'Total image issues are: {tot_img_issues}')

