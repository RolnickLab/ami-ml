#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date started  : April 12, 2024
About         : Testing a webdataset read
"""

import webdataset as wds
from itertools import islice


wds_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/webdataset/binary_classification/binary-000000.tar"

dataset = (wds.WebDataset(wds_file)
           .decode("pil")
           .to_tuple("jpg", "json"))

count = 0
for image, annotation in islice(dataset, 0, 3):
    image.save("/home/mila/a/aditya.jain/ami-ml/src/eccv2024/"+str(count)+".jpg") 
    print(annotation["label"])
    count += 1