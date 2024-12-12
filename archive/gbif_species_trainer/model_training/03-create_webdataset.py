#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Last modified : May 9th, 2023
About         : Creates a webdataset
"""

import json
import os
import random

from PIL import Image
from torchvision import transforms
import PIL
import numpy as np
import pandas as pd
import torch
import webdataset as wds
import argparse


def set_random_seed(random_seed: int):
    """set random seed for reproducibility"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def padding(image: PIL.Image.Image):
    """returns the padding transformation required based on image shape"""

    height, width = np.shape(image)[0], np.shape(image)[1]

    if height < width:
        pad_transform = transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        pad_transform = transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        return None

    return pad_transform


def create_webdataset(args):
    """creates dataset samples"""

    # user input variables
    dataset_dir = args.dataset_dir
    dataset_filepath = args.dataset_filepath
    label_filepath = args.label_filepath
    img_resize = args.image_resize
    webdataset_patern = args.webdataset_patern
    max_shard_size = args.max_shard_size
    random_seed = args.random_seed

    # set random seed
    set_random_seed(random_seed)

    # define other variables
    dataset_df = pd.read_csv(dataset_filepath)
    dataset_df = dataset_df.sample(frac=1)
    label_list = json.load(open(label_filepath))
    sink = wds.ShardWriter(webdataset_patern, maxsize=max_shard_size)
    corrupt_img = 0
    not_found_img = 0

    for _, row in dataset_df.iterrows():
        image_path = (
            dataset_dir
            + row["family"]
            + "/"
            + row["genus"]
            + "/"
            + row["species"]
            + "/"
            + row["filename"]
        )

        if not os.path.isfile(image_path):
            print(f"File {image_path} not found", flush=True)
            not_found_img += 1
            continue

        # check issue with image opening; completely corrupted
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
        except PIL.UnidentifiedImageError:
            print(f"Unidentified Image Error on file {image_path}", flush=True)
            corrupt_img += 1
            continue
        except OSError:
            print(f"OSError Error on file {image_path}", flush=True)
            corrupt_img += 1
            continue

        # image transforms
        padding_transform = padding(image)
        if padding_transform:
            image = padding_transform(image)
        resize_transform = transforms.Compose([transforms.Resize((img_resize, img_resize))])

        # check for partial image corruption
        try:
            image = resize_transform(image)
        except:
            print(f"Partial corruption of file {image_path}", flush=True)
            corrupt_img += 1
            continue

        # write the image
        fpath = (
            row["family"]
            + "/"
            + row["genus"]
            + "/"
            + row["species"]
            + "/"
            + row["filename"]
        )
        fpath = os.path.splitext(fpath)[0].lower()
        species_list = label_list["species"]
        label = row["species"]
        label = species_list.index(label)

        sample = {"__key__": fpath, "jpg": image, "cls": label}
        sink.write(sample)

    # done!
    sink.close()
    print(f"Webdataset creating completed.", flush=True)
    print(f"{corrupt_img} images are corrupted.", flush=True)
    print(f"{not_found_img} images are not found.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        help="root directory containing the gbif data",
        required=True,
    )
    parser.add_argument(
        "--dataset_filepath",
        help="file path containing every data point information",
        required=True,
    )
    parser.add_argument(
        "--label_filepath",
        help="file path containing numerical label information",
        required=True,
    )
    parser.add_argument(
        "--image_resize",
        help="resizing image to (size x size)",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--webdataset_patern",
        help="path and type to save the webdataset",
        required=True,
    )
    parser.add_argument(
        "--max_shard_size",
        help="the maximum shard size",
        default=100 * 1024 * 1024,
        type=int,
    )
    parser.add_argument(
        "--random_seed",
        help="random seed for reproducible experiments",
        default=42,
        type=int,
    )
    args = parser.parse_args()

    create_webdataset(args)
