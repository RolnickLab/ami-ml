#!/usr/bin/env python
# coding: utf-8

# In[1]:


""""
Author        : Aditya Jain
Date started  : May 20, 2022
About         : builds a video given the images and their localization, tracking information
"""

import cv2
import os
import time
import pandas as pd


### User defined variables

data_dir    = '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/'
image_dir   = data_dir + '2022_05_14/'
track_file  = data_dir + 'tracking_annotation-2022_05_14.csv'

data_images = os.listdir(image_dir)
# data_images.sort()
data_annot  = pd.read_csv(track_file)

# output video settings
test_img    = cv2.imread(image_dir + data_images[0])
height, width, layers = test_img.shape
vid_out     = cv2.VideoWriter(data_dir + '2022_05_14_localiz-tracking.mp4',
                              cv2.VideoWriter_fourcc(*'XVID'), 
                              framerate, 
                              (width,height))


# In[10]:


def prev_track_centre(annot_data, img_name, track_id):
    """returns centre given a track id and image, if available"""
    
    img_points = annot_data[annot_data['image']==img_name]
    
    for i in range(len(img_points)):
        if img_points.iloc[i,1]==track_id:
            return [img_points.iloc[i,6], img_points.iloc[i,7]]
        
    return [None, None]


# In[11]:


i = 0
prev_image_name = ''
start           = time.time()
img_count       = 0

while i<len(data_annot):
    image_name = data_annot.loc[i, 'image']
    image = cv2.imread(image_dir + image_name)
    img_count += 1
    
    while i<len(data_annot) and data_annot.loc[i, 'image']==image_name:
        cv2.rectangle(image,
                      (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']),
                      (data_annot.loc[i, 'bb_botright_x'], data_annot.loc[i, 'bb_botright_y']),
                      (0,0,255),
                       3)
        cv2.putText(image, 'ID: '+str(data_annot.loc[i, 'track_id']), 
                    (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0,0,255), 
                    2)        
        cv2.circle(image,
                   (data_annot.loc[i, 'bb_centre_x'], data_annot.loc[i, 'bb_centre_y']), 
                   4, 
                   (0,0,255), 
                   -1)
        
        # showing the previous track
        if prev_image_name:
            prev_centre = prev_track_centre(data_annot, prev_image_name, data_annot.loc[i, 'track_id'])
            if prev_centre[0]:
                cv2.line(image,
                         (prev_centre[0], prev_centre[1]),
                         (data_annot.loc[i, 'bb_centre_x'], data_annot.loc[i, 'bb_centre_y']),
                         (0,0,255), 
                          3)                
                
        i += 1    
    
    prev_image_name = image_name     
    vid_out.write(image)
        
vid_out.release()
print('Time take to build video in minutes: ', (time.time()-start)/60)
print('Total images: ', img_count)


# In[ ]:




