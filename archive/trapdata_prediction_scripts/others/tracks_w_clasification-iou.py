#!/usr/bin/env python
# coding: utf-8

# In[1]:


""""
Author        : Aditya Jain
Date started  : May 22, 2022
About         : given image sequences localization and classification info, builds the tracks
"""

import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cost_method.iou import intersection_over_union


# #### User-Defined Inputs

# In[2]:


data_dir  = '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Vermont/'

# cost thresholding for removing false tracks
COST_THR  = 1   


# #### Variable Definitions

# In[3]:


image_folder = '2022_05_13'
image_dir    = data_dir + image_folder + '/'
annot_file   = data_dir + 'localize_classify_annotation-' + image_folder + '.json'
track_file   = data_dir + 'track_localize_classify_annotation-' + image_folder + '.csv'

data_images = os.listdir(image_dir)
data_annot  = json.load(open(annot_file))

track_info  = []    # [<image_name>, <track_id>, <bb_topleft_x>, <bb_topleft_y>, <bb_botright_x>, <bb_botright_y>
                    #  <bb_centre_x>, <bb_centre_y>, <class>, <sub_class>, <confidence>]
track_id    = 1


# #### Tracking Part

# In[4]:


def find_track_id(image_name, annot):
    """finds the track id for a given image and annotation"""
    
    global track_info    
    idx = -1
    
    while True:
        if track_info[idx][0] == image_name:
            if track_info[idx][2:6] == annot:
                return track_info[idx][1]            
        idx -= 1
    
    
def save_track(data_images, data_annot, idx):
    """
    finds the track between annotations of two consecutive images
    
    Args:
    data_images (list) : list of trap images
    data_annot (dict)  : dictionary containing annotation information for each image
    idx (int)          : image index for which the track needs to be found
    """
    
    global track_info, track_id, COST_THR
    
    # finding the moth boxes
    image1_annot_data = data_annot[data_images[idx-1]]
    image1_annot      = []
    image1_meta_data  = []
    for i in range(len(image1_annot_data[0])):
        if image1_annot_data[2][i]=='moth':
            image1_annot.append(image1_annot_data[0][i])
            image1_meta_data.append([image1_annot_data[2][i],                                      image1_annot_data[3][i],                                      image1_annot_data[4][i]])
            
    image2_annot_data = data_annot[data_images[idx]]
    image2_annot      = []
    image2_meta_data  = []
    for i in range(len(image2_annot_data[0])):
        if image2_annot_data[2][i]=='moth':
            image2_annot.append(image2_annot_data[0][i])
            image2_meta_data.append([image2_annot_data[2][i],                                      image2_annot_data[3][i],                                      image2_annot_data[4][i]])
        else:
            track_info.append([data_images[idx], 'NA', 
                       image2_annot_data[0][i][0], image2_annot_data[0][i][1], 
                       image2_annot_data[0][i][2], image2_annot_data[0][i][3],
                       image2_annot_data[0][i][0] + int((image2_annot_data[0][i][2]-image2_annot_data[0][i][0])/2),
                       image2_annot_data[0][i][1] + int((image2_annot_data[0][i][3]-image2_annot_data[0][i][1])/2),
                       image2_annot_data[2][i], image2_annot_data[3][i], image2_annot_data[4][i]
                       ])
    
    cost_matrix  = np.zeros((len(image2_annot), len(image1_annot)))
    
    for i in range(len(image2_annot)):
        for j in range(len(image1_annot)):            
            iou              = intersection_over_union(image1_annot[j], image2_annot[i])
            cost             = 1-iou
            cost_matrix[i,j] = cost
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix) 
    
    row_ind = list(row_ind)
    col_ind = list(col_ind)
    
    for i in range(len(image2_annot)):
        # have a previous match
        if i in row_ind:          
            row_idx = row_ind.index(i)
            col_idx = col_ind[row_idx]
            
            # have a reasonable match from previous frame
            if cost_matrix[i, col_idx] < COST_THR:
                cur_id  = find_track_id(data_images[idx-1], image1_annot[col_idx])
                track_info.append([data_images[idx], cur_id, 
                               image2_annot[i][0], image2_annot[i][1],
                               image2_annot[i][2], image2_annot[i][3],
                               image2_annot[i][0] + int((image2_annot[i][2]-image2_annot[i][0])/2),
                               image2_annot[i][1] + int((image2_annot[i][3]-image2_annot[i][1])/2),
                               image2_meta_data[i][0], image2_meta_data[i][1], image2_meta_data[i][2]])
            
            # the cost of matching is too high; false match; thresholding; start a new track
            else:
                track_info.append([data_images[idx], track_id, 
                               image2_annot[i][0], image2_annot[i][1],
                               image2_annot[i][2], image2_annot[i][3],
                               image2_annot[i][0] + int((image2_annot[i][2]-image2_annot[i][0])/2),
                               image2_annot[i][1] + int((image2_annot[i][3]-image2_annot[i][1])/2),
                               image2_meta_data[i][0], image2_meta_data[i][1], image2_meta_data[i][2]])
                track_id += 1
                
        # no match, this is a new track 
        else:
            track_info.append([data_images[idx], track_id, 
                               image2_annot[i][0], image2_annot[i][1],
                               image2_annot[i][2], image2_annot[i][3],
                               image2_annot[i][0] + int((image2_annot[i][2]-image2_annot[i][0])/2),
                               image2_annot[i][1] + int((image2_annot[i][3]-image2_annot[i][1])/2),
                               image2_meta_data[i][0], image2_meta_data[i][1], image2_meta_data[i][2]])
            track_id += 1
    
    
def draw_bounding_boxes(image, annotation):
    """draws bounding box annotation for a given image"""

    for annot in annotation:
        cv2.rectangle(image,(annot[0], annot[1]),(annot[2], annot[3]),(0,0,255),3)
        
    return image
        


# #### Build the tracking annotation for the first image

# In[5]:


first_annot      = data_annot[data_images[0]][0]
first_annot_data = data_annot[data_images[0]]

for i in range(len(first_annot)):
    if first_annot_data[2][i]=='nonmoth':
        track_info.append([data_images[0], 'NA', 
                       first_annot[i][0], first_annot[i][1], 
                       first_annot[i][2], first_annot[i][3],
                       first_annot[i][0] + int((first_annot[i][2]-first_annot[i][0])/2),
                       first_annot[i][1] + int((first_annot[i][3]-first_annot[i][1])/2),
                       first_annot_data[2][i], first_annot_data[3][i], first_annot_data[4][i]
                       ])
    else:
        track_info.append([data_images[0], track_id, 
                       first_annot[i][0], first_annot[i][1], 
                       first_annot[i][2], first_annot[i][3],
                       first_annot[i][0] + int((first_annot[i][2]-first_annot[i][0])/2),
                       first_annot[i][1] + int((first_annot[i][3]-first_annot[i][1])/2),
                       first_annot_data[2][i], first_annot_data[3][i], first_annot_data[4][i]
                       ])
        track_id += 1


# #### Build the tracking annotation for the rest images 

# In[6]:


for i in range(1, len(data_images)):
    save_track(data_images, data_annot, i)

track_df = pd.DataFrame(track_info, columns =['image', 'track_id', 'bb_topleft_x', 
                                              'bb_topleft_y', 'bb_botright_x', 'bb_botright_y',
                                              'bb_centre_x', 'bb_centre_y', 'class', 'subclass', 
                                              'confidence'])

track_df.to_csv(track_file, index=False)
# track_df.to_csv('tracking_annotation.csv', index=False)


# #### Visualization - NOT to be run

# In[7]:


# img1 = cv2.imread(image_dir + data_images[i-1])
# img2 = cv2.imread(image_dir + data_images[i])
    
    
# img1 = draw_bounding_boxes(img1, data_annot[data_images[i-1]][0])
# cv2.imwrite('image1.jpg', img1)
# img2 = draw_bounding_boxes(img2, data_annot[data_images[i]][0])       
# cv2.imwrite('image2.jpg', img2)


# In[ ]:




