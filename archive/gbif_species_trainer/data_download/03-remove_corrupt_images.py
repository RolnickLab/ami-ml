#!/usr/bin/env python
# coding: utf-8

"""
Author	           : Aditya Jain
Date last modified : July 31, 2023
About              : Removes corrupted images from the database
"""

import argparse
import pandas as pd
import glob
import os

from PIL import Image
from multiprocessing import Pool
from torchvision import transforms


def delete_image(filepath: str):
    """deletes image based on pre-defined corruption metrics"""

    # sanity test to check if the image is resizable
    resize_transform = transforms.Compose([transforms.Resize((150, 150))])

    # check for complete or partial image corruption
    try:
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = resize_transform(image)
    except:
        print(f"{filepath} is corrupted and now deleted.", flush=True)
        os.remove(filepath)


def remove_corrupt_images(args: argparse.Namespace):
    """main function for removing the corrupt images"""

    species_list = pd.read_csv(args.species_checklist)

    print(f"Removing corrupted data for checklist: {args.species_checklist}")

    for _, row in species_list.iterrows():
        family = row["family"]
        genus = row["genus"]
        species = row["gbif_species"]

        # collect all image files for the species
        image_directory = args.data_directory + family + "/" + genus + "/" + species
        image_files = glob.glob(image_directory + "/*.jpg")

        # delete corrupted data using multi-processing
        if len(image_files) != 0:
            with Pool(processes=args.num_workers) as pool:
                pool.map(delete_image, image_files)

        print(f"Corruption check complete for {species}.", flush=True)

    print("Finished removing corrupted images!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_directory", help="root folder where image data is saved", required=True
    )

    parser.add_argument(
        "--species_checklist",
        help="csv file containing list of species names along with accepted taxon keys",
        required=True,
    )

    parser.add_argument(
        "--num_workers",
        help="number of CPUs available",
        type=int,
    )

    args = parser.parse_args()
    remove_corrupt_images(args)
