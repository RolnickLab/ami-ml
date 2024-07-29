"""
Test script to check if webdataset conversion leads to loss of image quality
"""

import glob
import os
from pathlib import Path

# 3rd party packages
import dotenv
import webdataset as wds
from PIL import Image

# Load secrets and config from optional .env file
dotenv.load_dotenv()
FINE_TUNING_DIR = os.environ.get("FINE_TUNING_DIR")

# List of insect crop images
img_files = glob.glob(f"{FINE_TUNING_DIR}/assets/original_images/*.jpg")

# Write the crops to webdataset file
wds_pattern = f"{FINE_TUNING_DIR}/test-%06d.tar"
with wds.ShardWriter(wds_pattern, maxsize=50 * 1024 * 1024) as sink:
    for img_file in img_files:
        img_basename = Path(img_file).stem
        img = Image.open(img_file)  # DO NOT USE IMAGE OPEN; directly read the file
        img_wds = {
            "__key__": img_basename,
            "png": img,
        }
        sink.write(img_wds)

# Read the webdataset file, recover and save the images
recovered_imgs = wds.WebDataset(f"{FINE_TUNING_DIR}/test-000000.tar").decode("pil")
for imgs in recovered_imgs:
    img = imgs["png"]
    img.save(f"{FINE_TUNING_DIR}/assets/wds_recovered_images/{imgs['__key__']}.png")


# Check if the recovered image is the same as the original image
