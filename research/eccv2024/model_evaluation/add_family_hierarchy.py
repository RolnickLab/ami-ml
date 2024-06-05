"""
Author: Aditya Jain
Date last modified: January 15, 2023
About: Add family info for subfamily, tribe and subtribe
"""

import json
import os

from pygbif import species as species_api


def add_family_level(data_dir: str, taxon_file: str):
    """Add family info for subfamily, tribe and subtribe"""

    # Get the image list and associated predctions
    pred_dir = os.path.join(data_dir, "ami_traps_dataset", "model_predictions")
    image_pred_list = os.listdir(pred_dir)

    # Load existing family taxon file
    hierarchy = ["SUBFAMILY", "TRIBE", "SUBTRIBE"]
    taxon_data = json.load(open(taxon_file))

    # Iterate over each image predictions
    for image_pred in image_pred_list:
        pred_data = json.load(open(os.path.join(pred_dir, image_pred)))

        # Iterate over each bounding box
        for bbox in pred_data:
            gt_label = bbox["ground_truth"][0]
            gt_rank = bbox["ground_truth"][1]

            if gt_rank in hierarchy:
                if gt_label not in taxon_data.keys():
                    gbif_lookup = species_api.name_backbone(name=gt_label)
                    family = gbif_lookup["family"]
                    taxon_data[gt_label] = family

    # Write the final data to disk
    with open(taxon_file, "w") as f:
        json.dump(taxon_data, f, indent=2)


if __name__ == "__main__":
    data_dir = "/home/mila/a/aditya.jain/scratch/cvpr2024_data"
    taxon_file = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/family-hierarchy.json"
    add_family_level(data_dir, taxon_file)
