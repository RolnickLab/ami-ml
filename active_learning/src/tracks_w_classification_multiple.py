#!/usr/bin/env python
# coding: utf-8

""""
Author        : Aditya Jain
Date started  : July 18, 2022
About         : Given image sequences localization and classification info, builds the tracks using CNN features, IoU, distance and box ratio
"""

import argparse
import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchsummary import summary
from torchvision import transforms, utils
from utils.cost_method.cnn_iou_dist_boxratio import TrackingCost
from utils.resnet50 import Resnet50


def find_track_id(image_name, annot):
    """finds the track id for a given image and annotation"""

    global track_info
    idx = -1

    while True:
        if track_info[idx][0] == image_name:
            if track_info[idx][2:6] == annot:
                return track_info[idx][1]
        idx -= 1


def save_track(image_dir, data_images, data_annot, idx, model, img_resize, device):
    """
    finds the track between annotations of two consecutive images

    Args:
    image_dir (str)    : path to image directory
    data_images (list) : list of trap images
    data_annot (dict)  : dictionary containing annotation information for each image
    idx (int)          : image index for which the track needs to be found
    model              : model for finding the cnn features
    img_resize (int)   : resizing size
    device (str)       : device being used, cuda/cpu
    """

    global track_info, track_id

    image1 = cv2.imread(image_dir + data_images[idx - 1])
    image2 = cv2.imread(image_dir + data_images[idx])
    img_shape = image1.shape
    img_diagonal = math.sqrt(img_shape[0] ** 2 + img_shape[1] ** 2)

    # finding the moth boxes
    image1_annot_data = data_annot[data_images[idx - 1]]
    image1_annot = []
    image1_meta_data = []
    for i in range(len(image1_annot_data[0])):
        if image1_annot_data[2][i] == "moth":
            image1_annot.append(image1_annot_data[0][i])
            image1_meta_data.append(
                [
                    image1_annot_data[2][i],
                    image1_annot_data[3][i],
                    image1_annot_data[4][i],
                ]
            )

    image2_annot_data = data_annot[data_images[idx]]
    image2_annot = []
    image2_meta_data = []
    for i in range(len(image2_annot_data[0])):
        if image2_annot_data[2][i] == "moth":
            image2_annot.append(image2_annot_data[0][i])
            image2_meta_data.append(
                [
                    image2_annot_data[2][i],
                    image2_annot_data[3][i],
                    image2_annot_data[4][i],
                ]
            )
        else:
            track_info.append(
                [
                    data_images[idx],
                    "NA",
                    image2_annot_data[0][i][0],
                    image2_annot_data[0][i][1],
                    image2_annot_data[0][i][2],
                    image2_annot_data[0][i][3],
                    image2_annot_data[0][i][0]
                    + int(
                        (image2_annot_data[0][i][2] - image2_annot_data[0][i][0]) / 2
                    ),
                    image2_annot_data[0][i][1]
                    + int(
                        (image2_annot_data[0][i][3] - image2_annot_data[0][i][1]) / 2
                    ),
                    image2_annot_data[2][i],
                    image2_annot_data[3][i],
                    image2_annot_data[4][i],
                ]
            )

    cost_matrix = np.zeros((len(image2_annot), len(image1_annot)))

    # building the cost matrix
    for i in range(len(image2_annot)):
        for j in range(len(image1_annot)):
            # getting image 2 cropped photo
            img2_annot = image2_annot[i]
            img2_moth = image2[
                img2_annot[1] : img2_annot[3], img2_annot[0] : img2_annot[2]
            ]
            img2_moth = Image.fromarray(img2_moth)

            # getting image1 cropped moth photo
            img1_annot = image1_annot[j]
            img1_moth = image1[
                img1_annot[1] : img1_annot[3], img1_annot[0] : img1_annot[2]
            ]
            img1_moth = Image.fromarray(img1_moth)

            tracking_cost = TrackingCost(
                img1_moth,
                img2_moth,
                image1_annot[j],
                image2_annot[i],
                model,
                img_diagonal,
                weights,
                img_resize,
                device,
            )
            cost_matrix[i, j] = tracking_cost.final_cost()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    row_ind = list(row_ind)
    col_ind = list(col_ind)

    # finding and relating the matches
    for i in range(len(image2_annot)):
        # have a previous match
        if i in row_ind:
            row_idx = row_ind.index(i)
            col_idx = col_ind[row_idx]

            # have a reasonable match from previous frame
            if cost_matrix[i, col_idx] < COST_THR:
                cur_id = find_track_id(data_images[idx - 1], image1_annot[col_idx])
                track_info.append(
                    [
                        data_images[idx],
                        cur_id,
                        image2_annot[i][0],
                        image2_annot[i][1],
                        image2_annot[i][2],
                        image2_annot[i][3],
                        image2_annot[i][0]
                        + int((image2_annot[i][2] - image2_annot[i][0]) / 2),
                        image2_annot[i][1]
                        + int((image2_annot[i][3] - image2_annot[i][1]) / 2),
                        image2_meta_data[i][0],
                        image2_meta_data[i][1],
                        image2_meta_data[i][2],
                    ]
                )

            # the cost of matching is too high; false match; thresholding; start a new track
            else:
                track_info.append(
                    [
                        data_images[idx],
                        track_id,
                        image2_annot[i][0],
                        image2_annot[i][1],
                        image2_annot[i][2],
                        image2_annot[i][3],
                        image2_annot[i][0]
                        + int((image2_annot[i][2] - image2_annot[i][0]) / 2),
                        image2_annot[i][1]
                        + int((image2_annot[i][3] - image2_annot[i][1]) / 2),
                        image2_meta_data[i][0],
                        image2_meta_data[i][1],
                        image2_meta_data[i][2],
                    ]
                )
                track_id += 1

        # no match, this is a new track
        else:
            track_info.append(
                [
                    data_images[idx],
                    track_id,
                    image2_annot[i][0],
                    image2_annot[i][1],
                    image2_annot[i][2],
                    image2_annot[i][3],
                    image2_annot[i][0]
                    + int((image2_annot[i][2] - image2_annot[i][0]) / 2),
                    image2_annot[i][1]
                    + int((image2_annot[i][3] - image2_annot[i][1]) / 2),
                    image2_meta_data[i][0],
                    image2_meta_data[i][1],
                    image2_meta_data[i][2],
                ]
            )
            track_id += 1


def draw_bounding_boxes(image, annotation):
    """draws bounding box annotation for a given image"""

    for annot in annotation:
        cv2.rectangle(image, (annot[0], annot[1]), (annot[2], annot[3]), (0, 0, 255), 3)

    return image


def tracking(args):
    """main for function for performing tracking"""

    global track_info, track_id, COST_THR, weights

    # User-defined Inputs
    data_dir = args.data_dir
    image_folder = args.image_folder
    COST_THR = (
        10000  # cost thresholding for removing false tracks: no rejection right now
    )
    WGT_CNN = args.weight_cnn  # weight on cnn features
    WGT_IOU = args.weight_iou  # weight on iou
    WGT_BOX = args.weight_box_ratio  # weight on box ratio
    WGT_DIS = args.weight_distance  # weight on distance ratio
    weights = [WGT_CNN, WGT_IOU, WGT_BOX, WGT_DIS]

    # Loading the tracking model
    image_resize = args.image_resize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.category_map_moth, "r") as f:
        categories_map = json.load(f)
    total_species = len(categories_map)
    model = Resnet50(total_species).to(device)
    path = args.model_moth_cnn
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = nn.Sequential(
        *list(model.children())[:-1]
    )  # only getting the last feature layer

    # Variable Definitions
    image_dir = data_dir + image_folder + "/"
    annot_file = data_dir + "localize_classify_annotation-" + image_folder + ".json"
    track_file = (
        data_dir + "track_localize_classify_annotation-" + image_folder + ".csv"
    )
    data_images = os.listdir(image_dir)
    data_annot = json.load(open(annot_file))
    track_info = (
        []
    )  # [<image_name>, <track_id>, <bb_topleft_x>, <bb_topleft_y>, <bb_botright_x>, <bb_botright_y>
    #  <bb_centre_x>, <bb_centre_y>, <class>, <sub_class>, <confidence>]
    track_id = 1

    # Build the tracking annotation for the first image
    first_annot = data_annot[data_images[0]][0]
    first_annot_data = data_annot[data_images[0]]

    for i in range(len(first_annot)):
        if first_annot_data[2][i] == "nonmoth":
            track_info.append(
                [
                    data_images[0],
                    "NA",
                    first_annot[i][0],
                    first_annot[i][1],
                    first_annot[i][2],
                    first_annot[i][3],
                    first_annot[i][0]
                    + int((first_annot[i][2] - first_annot[i][0]) / 2),
                    first_annot[i][1]
                    + int((first_annot[i][3] - first_annot[i][1]) / 2),
                    first_annot_data[2][i],
                    first_annot_data[3][i],
                    first_annot_data[4][i],
                ]
            )
        else:
            track_info.append(
                [
                    data_images[0],
                    track_id,
                    first_annot[i][0],
                    first_annot[i][1],
                    first_annot[i][2],
                    first_annot[i][3],
                    first_annot[i][0]
                    + int((first_annot[i][2] - first_annot[i][0]) / 2),
                    first_annot[i][1]
                    + int((first_annot[i][3] - first_annot[i][1]) / 2),
                    first_annot_data[2][i],
                    first_annot_data[3][i],
                    first_annot_data[4][i],
                ]
            )
            track_id += 1

    # Build the tracking annotation for the rest images
    for i in range(1, len(data_images)):
        save_track(image_dir, data_images, data_annot, i, model, image_resize, device)

    # Saving the tracking information
    track_df = pd.DataFrame(
        track_info,
        columns=[
            "image",
            "track_id",
            "bb_topleft_x",
            "bb_topleft_y",
            "bb_botright_x",
            "bb_botright_y",
            "bb_centre_x",
            "bb_centre_y",
            "class",
            "subclass",
            "confidence",
        ],
    )
    track_df.to_csv(track_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", help="root directory containing the trap data", required=True
    )
    parser.add_argument(
        "--image_folder",
        help="date folder within root directory containing the images",
        required=True,
    )
    parser.add_argument(
        "--model_moth_cnn",
        help="path to the moth model for comparison of cnn features",
        required=True,
    )
    parser.add_argument(
        "--image_resize",
        help="resizing image for model prediction",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--category_map_moth",
        help="path to the moth category map for converting integer labels to name labels",
        required=True,
    )
    parser.add_argument(
        "--weight_cnn", help="weight factor on the cnn features", default=1, type=int
    )
    parser.add_argument(
        "--weight_iou",
        help="weight factor on intersection over union",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--weight_box_ratio",
        help="weight factor on ratio of the box areas",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--weight_distance",
        help="weight factor on the distance between boxes",
        default=1,
        type=int,
    )
    args = parser.parse_args()
    tracking(args)
