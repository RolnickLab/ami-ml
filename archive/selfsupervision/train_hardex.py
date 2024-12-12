#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Author        : Aditya Jain
Date started  : 10th January, 2022
About         : This script fine tunes the model with hard examples
'''


# In[2]:


from comet_ml import Experiment

experiment = Experiment(
    api_key="epeaAhyRcHSkn92H4kusmbX8k",
    project_name="selfsupervision",
    workspace="adityajain07",
)


# In[2]:


import torchvision
from torchvision import models
from torchvision import datasets
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from data.imagenetvaldataset import ImagenetValDataset
from data.hardexdataset import HardExDataset

import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import argparse

# #### Loading pre-trained ResNet50 Model

# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model  = models.resnet50(pretrained=True).to(device)


# #### Edits to make

# In[4]:

parser             = argparse.ArgumentParser()
parser.add_argument("--data_path", help = "directory containing the data")
args               = parser.parse_args()

val_root_dir       = args.data_path + '/' + 'val/'
val_label_list     = '/home/mila/a/aditya.jain/mothAI/selfsupervision/data/validation_imagenet_labels.csv'
val_convert_list   = '/home/mila/a/aditya.jain/mothAI/selfsupervision/data/imagenet_modified_labels.csv'

hardex_data_dir    = args.data_path + '/' + 'hard_examples/'
hardex_label_list  = '/home/mila/a/aditya.jain/mothAI/selfsupervision/data/hard_examples_data.csv'

image_resize       = 224
batch_size         = 32

DTSTR              = datetime.datetime.now()
DTSTR              = DTSTR.strftime("%Y-%m-%d-%H-%M")
mod_save_path      = '/home/mila/a/aditya.jain/logs/'
mod_name           = 'selfsupervisemodel'
mod_ver            = 'v1'
save_path          = mod_save_path + mod_name + '_' + mod_ver  + '_' + DTSTR + '.pt'
early_stop         = 4
epochs             = 30

transformer        = transforms.Compose([
                        transforms.Resize((image_resize, image_resize)),              # resize the image to 224x224 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])


# #### Evaluation on Validation Data

# In[9]:


val_data          = ImagenetValDataset(val_root_dir, val_label_list, val_convert_list, transformer)
val_dataloader    = DataLoader(val_data,batch_size=batch_size)


# In[7]:


def batch_accuracy(predictions, labels):
    '''
    calculates top1, top5 and top10 correct predictions in a batch
    '''
    top1         = 0
    top5         = 0
    top10        = 0
    
    _, pr_indices1  = torch.topk(predictions, 1)
    _, pr_indices5  = torch.topk(predictions, 5)
    _, pr_indices10 = torch.topk(predictions, 10)
    
    for i in range(len(labels)):
        if labels[i] in pr_indices1[i]:
            top1 += 1
        
        if labels[i] in pr_indices5[i]:
            top5 += 1
            
        if labels[i] in pr_indices10[i]:
            top10 += 1
            
    return top1, top5, top10
    


# In[8]:


top1_correct  = 0
top5_correct  = 0
top10_correct = 0
total         = 0

model.eval()
for image_batch, label_batch in val_dataloader:
    
    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
    predictions              = model(image_batch)    
    top1, top5, top10        = batch_accuracy(predictions, label_batch)
   
    top1_correct    += top1
    top5_correct    += top5
    top10_correct   += top10
    total           += len(label_batch)
    
pretrain_accuracy                   = {}
pretrain_accuracy['top1_pretrain_val']  = top1_correct/total*100
pretrain_accuracy['top5_pretrain_val']  = top5_correct/total*100
pretrain_accuracy['top10_pretrain_val'] = top10_correct/total*100

experiment.log_metrics(pretrain_accuracy)


# #### Fine-tuning using hard examples

# In[6]:


hardex_data        = HardExDataset(hardex_data_dir, hardex_label_list, transformer)
hardex_dataloader  = DataLoader(hardex_data, batch_size=batch_size, shuffle=True)


# In[7]:


loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[10]:


start_val_loss  = 10000000
lowest_val_loss = start_val_loss
early_stp_count = 0

for epoch in range(epochs):
    train_loss = 0
    val_loss   = 0
    s_time     = time.time()
    
    # model fine-tuning
    model.train()
    for image_batch, label_batch in hardex_dataloader:
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        label_batch              = label_batch.squeeze_() 
        
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs   = model(image_batch)        
        t_loss    = loss_func(outputs, label_batch)
        t_loss.backward()
        optimizer.step()        
        train_loss += t_loss.item()
        
    experiment.log_metric("loss_train", train_loss, epoch=epoch)
    
    # model evaluation
    top1_correct  = 0
    top5_correct  = 0
    top10_correct = 0
    total         = 0

    model.eval()
    for image_batch, label_batch in val_dataloader:    
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        label_batch              = label_batch.squeeze_() 
        predictions              = model(image_batch)
        
        v_loss    = loss_func(predictions, label_batch)
        val_loss += v_loss.item()
        
        top1, top5, top10  = batch_accuracy(predictions, label_batch)   
        top1_correct      += top1
        top5_correct      += top5
        top10_correct     += top10
        total             += len(label_batch)
    
    experiment.log_metric("loss_val", val_loss, epoch=epoch)
    experiment.log_metric('top1_tuned_val', top1_correct/total*100, epoch=epoch)
    experiment.log_metric('top5_tuned_val', top5_correct/total*100, epoch=epoch)
    experiment.log_metric('top10_tuned_val', top10_correct/total*100, epoch=epoch)
    
    e_time = (time.time()-s_time)/60      
    experiment.log_metric("time_per_epoch", e_time, epoch=epoch)
    
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
        
    if early_stp_count>=early_stop:
        break 


# In[ ]:


experiment.end()


# In[ ]:




