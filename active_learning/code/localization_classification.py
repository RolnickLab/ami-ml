#!/usr/bin/env python
# coding: utf-8

"""
Author: Aditya Jain
Date last modified: July 11, 2023
About: Localization and classification on raw trap images, and saving the annotation information
"""

import torch
import torchvision.models as torchmodels
import torchvision
import os
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import timm
import argparse
from torchvision.utils import save_image

from resnet50 import Resnet50
from classification import ClassificationInference


def padding(image: Image.Image):
    """padding transformation required based on image shape"""

    height, width = np.shape(image)[0], np.shape(image)[1]

    if height < width:
        pad_transform = transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        pad_transform = transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        return None

    return pad_transform


def localization_classification(args: argparse.Namespace):
    """main function for localization and classification"""

    data_dir = args.data_dir
    image_folder = args.image_folder
    data_path = data_dir + image_folder + "/"
    save_path = data_dir
    annot_file = "localize_classify_annotation-" + image_folder + ".json"

    # Get cpu or gpu device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load localization model
    model_localize = torchvision.models.detection.__dict__[args.model_localize_type](
        num_classes=2
    )
    model_path = args.model_localize
    checkpoint = torch.load(model_path, map_location=device)
    model_localize.load_state_dict(checkpoint["model_state_dict"])
    model_localize = model_localize.to(device)
    model_localize.eval()

    # Load binary classification model (moth / non-moth)
    category_map_json = args.category_map_moth_nonmoth
    model_path = args.model_moth_nonmoth
    model_type = args.model_moth_nonmoth_type
    model_binary = ClassificationInference(
        model_path, model_type, category_map_json, device, args.image_resize
    )

    # Load moth classification model
    category_map_json = args.category_map_moth
    model_path = args.model_moth
    model_type = args.model_moth_type
    model_moth = ClassificationInference(
        model_path, model_type, category_map_json, device, args.image_resize
    )

    # Main loop for prediction on trap data
    annot_data = {}
    SCORE_THR = args.localize_score_thresh / 100
    transform = transforms.Compose([transforms.ToTensor()])
    image_list = os.listdir(data_path)
    assert len(image_list)>0, 'No image files read.'

    for img in image_list:
        if not img.endswith(".jpg"):
            continue
        image_path = data_path + img
        try:
            raw_image = Image.open(image_path)
        except:
            print(f"Issue with image {image_path}")
            continue
        image_size = np.shape(raw_image)
        total_area = image_size[0] * image_size[1]
        image = transform(raw_image)
        image_pred = torch.unsqueeze(image, 0).to(device)
        output = model_localize(image_pred)
        bboxes = output[0]["boxes"][output[0]["scores"] > SCORE_THR]

        bbox_list = []
        label_list = []
        class_list = []  # moth / non-moth
        subclass_list = []  # moth species / non-moth
        conf_list = []  # confidence list
        area_list = []  # percentage area of the sheet

        for box in bboxes:
            box_numpy = box.detach().cpu().numpy()
            bbox_list.append(
                [
                    int(box_numpy[0]),
                    int(box_numpy[1]),
                    int(box_numpy[2]),
                    int(box_numpy[3]),
                ]
            )
            label_list.append(1)
            insect_area = (int(box_numpy[2]) - int(box_numpy[0])) * (
                int(box_numpy[3]) - int(box_numpy[1])
            )
            area_per = int(insect_area / total_area * 100)

            cropped_image = image[
                :,
                int(box_numpy[1]) : int(box_numpy[3]),
                int(box_numpy[0]) : int(box_numpy[2]),
            ]
            transform_to_PIL = transforms.ToPILImage()
            cropped_image = transform_to_PIL(cropped_image)
            if np.shape(cropped_image)[2] > 3:
                cropped_image = cropped_image.convert("RGB")

            padding_transform = padding(cropped_image)
            if padding_transform:
                cropped_image = padding_transform(cropped_image)

            # prediction for moth / non-moth
            categ, conf = model_binary.predict(cropped_image, confidence=True)
            if categ == "nonmoth":
                class_list.append("nonmoth")
                subclass_list.append("nonmoth")
                conf_list.append(int(conf * 100))
                area_list.append(area_per)
            else:
                categ, conf = model_moth.predict(cropped_image, confidence=True)
                class_list.append("moth")
                subclass_list.append(categ)
                conf_list.append(int(conf * 100))
                area_list.append(area_per)

        annot_data[img] = [bbox_list, label_list, class_list, subclass_list, conf_list]

    with open(save_path + annot_file, "w") as outfile:
        json.dump(annot_data, outfile)


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
        "--image_resize", help="image resize for classification", default=224, type=int
    )
    parser.add_argument(
        "--model_localize_type", help="pytorch object detector type", required=True
    )
    parser.add_argument(
        "--model_localize", help="path to the localization model", required=True
    )
    parser.add_argument(
        "--localize_score_thresh",
        help="confidence threshold of the object detector over which to consider predictions",
        default=99,
        type=int,
    )
    parser.add_argument(
        "--model_moth",
        help="path to the fine-grained moth classification model",
        required=True,
    )
    parser.add_argument(
        "--model_moth_type",
        help="the type of model used; resnet50 or tf_efficientnetv2_b3",
        required=True,
    )
    parser.add_argument(
        "--model_moth_nonmoth", help="path to the moth-nonmoth model", required=True
    )
    parser.add_argument(
        "--model_moth_nonmoth_type",
        help="the type of model used; resnet50 or tf_efficientnetv2_b3",
        required=True,
    )
    parser.add_argument(
        "--category_map_moth",
        help="path to the moth category map for converting integer labels to name labels",
        required=True,
    )
    parser.add_argument(
        "--category_map_moth_nonmoth",
        help="path to the moth-nonmoth category map for converting integer labels to name labels",
        required=True,
    )
    args = parser.parse_args()
    localization_classification(args)
