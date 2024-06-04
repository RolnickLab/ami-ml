"""
Author: Aditya Jain
Date last modified: October 25, 2023
About: Download raw images for the annotated tasks
"""

import json
import os
import urllib.error
import urllib.request

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()

ECCV2024_DATA = os.getenv("ECCV2024_DATA_PATH")

annotation_file = f"{ECCV2024_DATA}/annotated-tasks-20231106.json"
data = json.load(open(annotation_file))
image_folder = f"{ECCV2024_DATA}/ami_traps_dataset/images"

# Refetch select images
image_list = [
    "20220718030444-00-14.jpg",
    "20220822040414-00-174.jpg",
    "01-20220928215959-snapshot.jpg",
]

for i in range(len(data)):
    image_url = data[i]["data"]["image"]
    image_name = os.path.basename(os.path.normpath(image_url))

    # Test code for a select images
    if image_name in image_list:
        try:
            urllib.request.urlretrieve(image_url, image_folder + "/" + image_name)
        except urllib.error.URLError as e:
            print(f"Error fetching {image_name}. The error is: {e}.")
