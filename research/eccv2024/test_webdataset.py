#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date started  : April 12, 2024
About         : Testing a webdataset read
"""

import os
from itertools import islice

import webdataset as wds

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()

ECCV2024_DATA = os.getenv("ECCV2024_DATA")
if ECCV2024_DATA is None:
    print("Please set ECCV2024_DATA in .env file", flush=True)
    exit(1)
filename = "binary-000000.tar"
wds_file = (
    f"{ECCV2024_DATA}/camera_ready_amitraps/webdataset/binary_classification/{filename}"
)

dataset = wds.WebDataset(wds_file).decode("pil").to_tuple("jpg", "json")

count = 0
for image, annotation in islice(dataset, 0, 3):
    image.save("./" + str(count) + ".jpg")
    print(annotation["label"])
    count += 1
    break
