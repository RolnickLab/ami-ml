#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date started  : April 9, 2024
About         : Exports the AMI-Traps dataset to webdataset format
"""

import os
import json
from PIL import Image
import webdataset as wds
import pandas as pd

def get_synonym(sp_checklist: pd.DataFrame, species: str):
    """Return synonym name of a species, if it exists on GBIF"""
    
    species_list.loc[species_list["gbif_species"] == species]["search_species"].values[0]


def export_to_webdataset(data_dir: str, export_dir: str, sp_checklist: pd.DataFrame):
    """Main function for exporting AMI-Traps to webdataset"""

    # Get the list of raw camera trap images and other metadata
    image_dir = os.path.join(data_dir, "images")
    metadata_dir = os.path.join(data_dir, "metadata")
    labels_dir = os.path.join(data_dir, "labels")
    image_list = os.listdir(image_dir)

    # Get ground truth class names
    gt_class_list = json.load(open(os.path.join(data_dir, "notes.json")))["categories"]

    # WebDataset specific variables
    max_shard_size = 50 * 1024 * 1024
    wbd_pattern_binary = os.path.join(
        export_dir, "binary_classification", "binary-%06d.tar"
    )
    wbd_pattern_fgrained = os.path.join(
        export_dir, "finegrained_classification", "fgrained-%06d.tar"
    )
    sink_binary = wds.ShardWriter(wbd_pattern_binary, maxsize=max_shard_size)
    sink_fgrained = wds.ShardWriter(wbd_pattern_fgrained, maxsize=max_shard_size)

    # Iterate over each image separately
    for image in image_list:
        img_basename = os.path.splitext(image)[0]

        # Fetch the region name for the image
        metadata_file = os.path.join(metadata_dir, img_basename + ".json")
        metadata = json.load(open(metadata_file))
        region = metadata["region"]

        # Read the raw image
        try:
            raw_image = Image.open(os.path.join(image_dir, image))
            img_width, img_height = raw_image.size
        except:
            raise Exception(f"Issue with image {image}")

        # Get the ground truth bounding box and label
        labels = open(os.path.join(labels_dir, img_basename + ".txt"), "r")

        # Iterate over each annotation/crop separately
        binary_crop_count = 1
        fgrained_crop_count = 1
        for line in labels:
            label_id, x, y, w, h = (
                int(line.split()[0]),
                float(line.split()[1]),
                float(line.split()[2]),
                float(line.split()[3]),
                float(line.split()[4]),
            )

            # Get the ground truth class name
            for class_entry in gt_class_list:
                if class_entry["id"] == label_id:
                    label_name = class_entry["name"]
                    label_rank = class_entry["rank"]
                    break

            # Ignore unlabeled crops
            if label_name != "Unidentifiable" and label_name != "Unclassified":
                # Get the insect crop
                x_start = int((x - w / 2) * img_width)
                y_start = int((y - h / 2) * img_height)
                w_px, h_px = int(w * img_width), int(h * img_height)
                insect_crop = raw_image.crop(
                    (x_start, y_start, x_start + w_px, y_start + h_px)
                )

                # Export to webdataset for binary classification
                sample_binary_annotation = {
                    "label": label_name, 
                    "region": region
                }
                sample_binary = {
                    "__key__": img_basename + "_" + str(binary_crop_count),
                    "jpg": insect_crop,
                    "json": sample_binary_annotation,
                }
                binary_crop_count += 1
                sink_binary.write(sample_binary)

                # Export to webdataset for finegrained classification, if moth crop
                if label_rank != "NA":

                    # If exists, get the synonym name
                    synonym = get_synonym(sp_checklist, label_name)

                    sample_fgrained_annotation = {
                        "label": label_name, 
                        "synonym": ...,
                        "gbif_id": ...,
                        "region": region
                    }
                    sample_fgrained = {
                        "__key__": img_basename + "_" + str(fgrained_crop_count),
                        "jpg": insect_crop,
                        "json": sample_fgrained_annotation,
                    }
                    fgrained_crop_count += 1
                    sink_fgrained.write(sample_fgrained)

    sink_binary.close()
    sink_fgrained.close()
    print(f"The export of the AMI-Traps dataset to webdataset is complete!", flush=True)


if __name__ == "__main__":
    data_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/ami_traps_dataset"
    export_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/webdataset"
    species_checklist = "/home/mila/a/aditya.jain/mothAI/species_lists/quebec-vermont-uk-denmark-panama_checklist_20231124.csv"
    species_list = pd.read_csv(species_checklist)

    export_to_webdataset(data_dir, export_dir, species_list)