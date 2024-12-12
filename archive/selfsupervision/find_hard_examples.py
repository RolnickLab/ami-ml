#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Author        : Aditya Jain
Date started  : 8th November, 2021
About         : This script is used for finding the hard examples using probability density
'''
import torchvision.models as models
from torch import nn
import torch
import cv2
import numpy as np
from torchvision import transforms, utils
from PIL import Image
import plotly.express as px
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help = "directory containing the data")
args   = parser.parse_args()

# #### Loading pre-trained ImageNet Model 

# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_mod = models.resnet50(pretrained=True)
resnet_mod = resnet_mod.to(device)
resnet_mod.eval()
print(device)


# #### Loading ImageNet Classes

# In[3]:


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# #### Video Loading and Processing

# In[6]:


def model_prediction(image, model, transformer, categories, device):
    '''
    returns the ImageNet class label for a model's prediction on an image
    '''
    softmax               = nn.Softmax(dim=1)
    
    image                 = Image.fromarray(image)
    image                 = transformer(image)            
    image                 = torch.unsqueeze(image, 0).to(device)
    
    prediction            = model(image)
    pred_softmax          = softmax(prediction)
    pred_val, pred_indx   = torch.topk(pred_softmax, 1)
    index                 = pred_indx.detach().cpu().numpy()
    
    return pred_val.detach().cpu().numpy(), categories[int(index[0])]


def save_logic(prediction_list, cur_class, threshold):
    '''
    implements the logic if the current frame should be classified as a hard example
    Args:
        prediction_list (list)  : the list of window predictions, contains class and confidence
        cur_class (string)      : the label of the current frame
        threshold (float)       : min. threshold for the most popular class 
        
        returns (bool)     : if current frame is a hard example
    '''
    tot_items   = len(prediction_list)
    mid_elem    = tot_items//2
    other_class = prediction_list[0][0]    # classes apart from the central frame
    
    for i in range(tot_items):
        if i!=mid_elem:
            if prediction_list[i][0]!=other_class or prediction_list[i][1]<threshold:
                return False, ''
    
    if prediction_list[mid_elem][0]!=other_class:
        return True, other_class
    else:
        return False, ''
    

def hard_examples(video_path, model, save_loc, window, threshold, categories, device):
    '''
    given an input video, finds and saves the hard examples
    Args:
        video_path (string): path for the video to be evaluated
        model (torch model): model to be ran for evaluation
        save_loc (string)  : location for the saving of hard examples
        window (int)       : no of frames to check on either side
        
        returns            : saves hard examples and count of examples in a video
    '''
    softmax        = nn.Softmax(dim=1)
    img_size       = 224    
    transformer    = transforms.Compose([
                            transforms.Resize((img_size, img_size)),                            
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    vidcap      = cv2.VideoCapture(video_path)     
    fps         = vidcap.get(cv2.CAP_PROP_FPS)           #  FPS of the video 
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)   #  total frame count 
    frame_indx  = window                                 #  starts from the window offset
    
    with torch.no_grad():
        pred_list  = []
        while frame_indx < (frame_count-window):
            cur_class   = ''
            cur_image  = ''
            
            if pred_list==[]:
                for frame in range(frame_indx-window, frame_indx+window+1):           
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)    # setting which frame to get        
                    success, image = vidcap.read()
                
                    if success:
                        p_probab, p_class  = model_prediction(image, model, transformer, categories, device) 
                        pred_list.append([p_class, p_probab[0][0]])
                        
                        # getting the label and image for current frame    
                        if frame==frame_indx:
                            cur_class = p_class
                            cur_image = image               
            else:
                cur_elem  = len(pred_list)//2 + 1
                cur_class = pred_list[cur_elem][0]
                
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_indx)    # setting which frame to get        
                success, image = vidcap.read()                
                if success:
                    cur_image = image
                 
                pred_list      = pred_list[1:]     # don't need the first element now
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_indx+window)    # setting which frame to get        
                success, image = vidcap.read()                
                if success:
                    p_probab, p_class  = model_prediction(image, model, transformer, categories, device)
                    pred_list.append([p_class, p_probab[0][0]])                
            
            flag, correct_class = save_logic(pred_list, cur_class, threshold)           
            if flag:
                save_dir = save_loc + correct_class
                
                # making a directory; if needed
                try:    
                    os.makedirs(save_dir)                     
                except:
                    pass
                
                exist_count   = len(os.listdir(save_dir))    # count of existing files in the folder
                img_save_path = save_dir + '/' + str(exist_count+1) + '_' + cur_class + '.jpg'
                cv2.imwrite(img_save_path, cur_image)
                print('Found hard example: ', img_save_path)
                
            frame_indx += 1                  
            


# In[9]:


# folders       = ['travel_blogger', 'home_interior']
folders       = ['home decor']
# video_rem     = ['travel_3.mp4', 'travel_5.mp4', 'travel_11.mp4', 'travel_14.mp4', 'travel_19.mp4']
ROOT_PATH     = args.data_path + '/' + 'selfsupervise_data/'
SAVE_LOC      = '/home/mila/a/aditya.jain/scratch/selfsupervise_data/hard_examples_v2/'
WINDOW        = 2
THRESHOLD     = 0.5

for folder in folders:
    class_path = ROOT_PATH + folder
    vid_files  = os.listdir(class_path)
    for video in vid_files:
        if video.lower().endswith('.mp4'):
            video_path = class_path + '/' + video
            print('Running for: ', video_path)
            hard_examples(video_path, resnet_mod, SAVE_LOC, WINDOW, THRESHOLD, categories, device)

# ## only for one-off
# for folder in folders:
#     class_path = ROOT_PATH + folder
#     vid_files  = os.listdir(class_path)
#     for video in vid_files:
#         if video in video_rem:
#             video_path = class_path + '/' + video
#             print('Running for: ', video_path)
#             hard_examples(video_path, resnet_mod, SAVE_LOC, WINDOW, THRESHOLD, categories, device)


# In[ ]:




