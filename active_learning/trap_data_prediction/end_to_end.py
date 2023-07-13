#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date modified : July 11, 2023
About         : This file does everything end-to-end: localization, classification, and tracking
"""

import torch
import torchvision.models as torchmodels
import torchvision
import os
import math
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
from torch import nn
from torchvision import transforms, utils
from torchsummary import summary
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import time
import timm
import argparse

from localization_classification import localization_classification
from tracks_w_classification_multiple import tracking

# User-defined variables 
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help = "root directory containing the trap data", required=True)
parser.add_argument("--image_folder", help = "date folder within root directory containing the images", required=True)
parser.add_argument("--model_localize_type", help = "pytorch object detector type", required=True)
parser.add_argument("--model_localize", help = "path to the localization model", required=True)
parser.add_argument("--localize_score_thresh", help = "confidence threshold of the object detector over which to consider predictions", default=99, type=int)
parser.add_argument("--model_moth", help = "path to the fine-grained moth classification model", required=True)
parser.add_argument("--model_moth_type", help = "the type of model used; resnet50 or tf_efficientnetv2_b3", required=True)
parser.add_argument("--model_moth_nonmoth", help = "path to the moth-nonmoth model", required=True)
parser.add_argument("--model_moth_nonmoth_type", help = "the type of model used; resnet50 or tf_efficientnetv2_b3", required=True)
parser.add_argument("--category_map_moth", help = "path to the moth category map for converting integer labels to name labels", required=True)
parser.add_argument("--category_map_moth_nonmoth", help = "path to the moth-nonmoth category map for converting integer labels to name labels", required=True)
parser.add_argument("--model_moth_cnn", help = "path to the moth model for comparison of cnn features", required=True)
parser.add_argument("--image_resize", help = "resizing image for model prediction", default=224, type=int)
parser.add_argument("--weight_cnn", help = "weight factor on the cnn features", default=1, type=int)
parser.add_argument("--weight_iou", help = "weight factor on intersection over union", default=1, type=int)
parser.add_argument("--weight_box_ratio", help = "weight factor on ratio of the box areas", default=1, type=int)
parser.add_argument("--weight_distance", help = "weight factor on the distance between boxes", default=1, type=int)
parser.add_argument("--region", help = "name of the region", required=True)
args   = parser.parse_args()

if __name__ == '__main__':
	localization_classification(args)
	tracking(args)
