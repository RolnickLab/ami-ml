#!/usr/bin/env python
# coding: utf-8

# In[10]:


"""
Author        : Aditya Jain
Date Started  : May 5, 2022
About         : Finds and deletes the corrupted images in the dataset
"""


# In[11]:


import os
import glob
import cv2
import time


# #### In moths dataset

# In[ ]:


moth_data_dir    = '/home/mila/a/aditya.jain/scratch/GBIF_Data/moths/'
total_images     = 0
count            = 0
s_time           = time.time()

for family in os.listdir(moth_data_dir):
    if not family.endswith('.csv') and not family.endswith('.ipynb_checkpoints'):
        for genus in os.listdir(moth_data_dir + family):        
            for species in os.listdir(moth_data_dir + family + '/' + genus):
                path       = moth_data_dir + family + '/' + genus + '/' + species
                file_data  = glob.glob(path + '/*.jpg')
                
                for file in file_data:
                    total_images += 1
                    image      = cv2.imread(file)
                    
                    if image is None:
                        print(file)
                        count += 1
                        os.remove(file)

print('Total images in the moth dataset: ', total_images)                        
print('Total corrupted images found in the moth dataset: ', count)
print('Time taken (mins): ', (time.time()-s_time)/60)


# #### In non-moths dataset

# In[ ]:


total_images     = 0
count            = 0
nonmoth_data_dir = '/home/mila/a/aditya.jain/scratch/GBIF_Data/nonmoths/'
s_time           = time.time()

for order in os.listdir(nonmoth_data_dir):
    if not order.endswith('.csv') and not order.endswith('.ipynb_checkpoints'):
        
        file_data  = glob.glob(nonmoth_data_dir + order + '/*.jpg')
        
        for file in file_data:
            total_images += 1
            image      = cv2.imread(file)
            
            if image is None:
                print(file)
                count += 1
                os.remove(file)

print('Total images in the non-moth dataset: ', total_images)                        
print('Total corrupted images found in the non-moth dataset: ', count)
print('Time taken (mins): ', (time.time()-s_time)/60)

