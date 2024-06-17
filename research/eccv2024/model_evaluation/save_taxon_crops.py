"""
Author: Aditya Jain
Date last modified: February 22, 2024
About: Save crops for a specific taxonomic name
"""

import json
import os
from pathlib import Path

from numpy import random
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def _helper_save_crop(
    bbox: list[str],
    image_pred_file: str,
    data_dir: str,
    save_dir: str,
    taxon: str,
    img_counter: int,
):
    """Save the insect crop image on the disk"""

    # Directory for loading image
    image_dir = Path(data_dir) / "ami_traps_dataset" / "images"

    # Parse the image name
    image_name = image_pred_file.split("_")[0] + ".jpg"

    # Read the raw image
    try:
        raw_image = Image.open(image_dir / image_name)
        img_width, img_height = raw_image.size
    except OSError as e:
        print(f"Error {e} with image {image_name}")

    # Convert the raw image to tensor
    transform_totensor = transforms.Compose([transforms.ToTensor()])
    try:
        image = transform_totensor(raw_image)
    except OSError as e:
        print(f"Error {e} with image {image_name}")

    # Get the insect crop and save on the disk
    x, y = float(bbox[0]), float(bbox[1])
    w, h = float(bbox[2]), float(bbox[3])
    x_start = int((x - w / 2) * img_width)
    y_start = int((y - h / 2) * img_height)
    w_px, h_px = int(w * img_width), int(h * img_height)
    cropped_image = image[:, y_start : y_start + h_px, x_start : x_start + w_px]
    crop_name = taxon + "_" + str(img_counter + 1) + ".png"
    save_image(cropped_image, Path(save_dir) / crop_name)


def save_insect_crop(
    data_dir: str,
    save_dir: str,
    taxon: str,
    num_crops_reqd: int,
    prob_random: float = 0.5,
):
    """Main function for saving insect crops for a particular taxon"""

    # Get the image list and associated predctions
    pred_dir = Path(data_dir) / "ami_traps_dataset" / "model_predictions" / "baseline"
    image_pred_list = os.listdir(pred_dir)

    # Iterate over each image predictions
    imgs_saved = 0
    complete_flag = False
    for image_pred in image_pred_list:
        if not complete_flag:
            with open(pred_dir / image_pred, "r") as f:
                pred_data = json.load(f)

            # Iterate over each bounding box
            for bbox in pred_data:
                gt = bbox["ground_truth"][0]
                bbox_coord = bbox["bbox_coordinates"]

                # Save random crops for the taxon required
                if gt == taxon and random.rand() < prob_random:
                    _helper_save_crop(
                        bbox_coord, image_pred, data_dir, save_dir, taxon, imgs_saved
                    )
                    imgs_saved += 1

                # Stop if got the required crops
                if imgs_saved == num_crops_reqd:
                    complete_flag = True


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    save_dir = f"{ECCV2024_DATA}/model_evaluation/plots/long-tail_traps_images"
    taxon = "Euchoeca nebulata"
    num_crops_reqd = 10
    save_insect_crop(ECCV2024_DATA, save_dir, taxon, num_crops_reqd, 1)
