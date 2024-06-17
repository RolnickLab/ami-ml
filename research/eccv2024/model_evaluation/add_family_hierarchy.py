"""
Author: Aditya Jain
Date last modified: January 15, 2023
About: Add family info for subfamily, tribe and subtribe
"""

import json
import os
from pathlib import Path

from pygbif import species as species_api


def add_family_level(data_dir: str, taxon_file: str):
    """Add family info for subfamily, tribe and subtribe"""

    # Get the image list and associated predctions
    pred_dir = Path(data_dir) / "ami_traps_dataset" / "model_predictions"
    image_pred_list = os.listdir(pred_dir)

    # Load existing family taxon file
    hierarchy = ["SUBFAMILY", "TRIBE", "SUBTRIBE"]
    with open(taxon_file, "r") as f:
        taxon_data = json.load(f)

    # Iterate over each image predictions
    for image_pred in image_pred_list:
        with open(pred_dir / image_pred, "r") as f:
            pred_data = json.load(f)

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
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    taxon_file = f"{ECCV2024_DATA}/family-hierarchy.json"
    add_family_level(ECCV2024_DATA, taxon_file)
