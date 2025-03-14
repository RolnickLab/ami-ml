""""
Author        : Aditya Jain
Date created  : 15th March, 2022
About         : Finds cosine similarity for a bounding box pair images
"""

import json
import os

import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torchsummary import summary
from torchvision import transforms, utils


def transform_image(image, img_resize=224):
    """transforms the cropped moth images for model prediction"""

    transformer = transforms.Compose(
        [
            transforms.Resize((img_resize, img_resize)),  # resize the image to 224x224
            transforms.ToTensor(),
        ]
    )
    image = transformer(image)

    # RGBA image; extra alpha channel
    if image.shape[0] > 3:
        image = image[0:3, :, :]

    # grayscale image; converted to 3 channels r=g=b
    if image.shape[0] == 1:
        to_pil = transforms.ToPILImage()
        to_rgb = transforms.Grayscale(num_output_channels=3)
        to_tensor = transforms.ToTensor()
        image = to_tensor(to_rgb(to_pil(image)))

    return image


def l1_normalize(v):
    norm = np.sum(np.array(v))
    return v / norm


def cosine_similarity(img1_moth, img2_moth, model, device="cuda", img_resize=224):
    """
    Finds cosine similarity for a bounding box pair images

    Args
    image1: cropped moth image 1
    image2: cropped moth image 2

    Return
    -----------
    0<=float<=1
    """
    img2_moth = transform_image(img2_moth, img_resize)
    img2_moth = torch.unsqueeze(img2_moth, 0).to(device)

    img1_moth = transform_image(img1_moth, img_resize)
    img1_moth = torch.unsqueeze(img1_moth, 0).to(device)

    # getting model features for each image
    with torch.no_grad():
        img2_ftrs = model(img2_moth)
        img2_ftrs = img2_ftrs.view(-1, img2_ftrs.size(0)).cpu()
        img2_ftrs = img2_ftrs.reshape((img2_ftrs.shape[0],))
        img2_ftrs = l1_normalize(img2_ftrs)

        img1_ftrs = model(img1_moth)
        img1_ftrs = img1_ftrs.view(-1, img1_ftrs.size(0)).cpu()
        img1_ftrs = img1_ftrs.reshape((img1_ftrs.shape[0],))
        img1_ftrs = l1_normalize(img1_ftrs)

    cosine_sim = np.dot(img1_ftrs, img2_ftrs) / (
        np.linalg.norm(img1_ftrs) * np.linalg.norm(img2_ftrs)
    )
    assert 0 <= cosine_sim <= 1, "cosine similarity score out of bounds"

    return cosine_sim
