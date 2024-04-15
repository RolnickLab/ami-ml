"""
Author: Aditya Jain
Date last modified: October 25, 2023
About: Download raw images for the annotated tasks
"""

import json
import urllib.request
import os

annotation_file = (
    "/home/mila/a/aditya.jain/scratch/cvpr2024_data/annotated-tasks-20231106.json"
)
data = json.load(open(annotation_file))
image_folder = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/ami_traps_dataset/images"

# Refetch select images
image_list = ["20220718030444-00-14.jpg", "20220822040414-00-174.jpg", "01-20220928215959-snapshot.jpg"]

for i in range(len(data)):
    image_url = data[i]["data"]["image"]
    image_name = os.path.basename(os.path.normpath(image_url))

    # Test code for a select images
    if image_name in image_list:
        try:
            urllib.request.urlretrieve(image_url, image_folder + "/" + image_name)
        except Exception as e:
            print("Error fetching {image_name}. The error is: {e}.")
