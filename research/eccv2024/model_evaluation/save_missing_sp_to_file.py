"""
Author: Aditya Jain
Date started: April 20, 2024
About: Check and save AMI-Traps species missing in AMI-GBIF database
"""

import json
import os
import pickle

import pandas as pd
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()

ECCV2024_DATA = os.getenv("ECCV2024_DATA_PATH")


def save_missing_species_keys(insect_crops_dir: str, gbif_taxonomy_map: pd.DataFrame):
    """Main function to save missing species keys to disk"""

    # Variable definitions
    missing_sp = []
    missing_sp_fname = "ami-traps_sp_missing_in_ami-gbif.pickle"

    # Take the unique list of species keys in AMI-GBIF data
    gbif_sp_keys = gbif_taxonomy_map["speciesKey"].tolist()

    # Get all moth insect crops label information
    insect_labels = json.load(
        open(os.path.join(insect_crops_dir, "fgrained_labels.json"))
    )

    # Iterate over each moth crop
    for img_name in insect_labels.keys():
        # Get ground truth label information
        gt_acceptedTaxonKey = insect_labels[img_name]["acceptedTaxonKey"]

        if (
            gt_acceptedTaxonKey
            and gt_acceptedTaxonKey != -1
            and gt_acceptedTaxonKey not in gbif_sp_keys
            and gt_acceptedTaxonKey not in missing_sp
        ):
            missing_sp.append(gt_acceptedTaxonKey)

    print(f"{len(missing_sp)} AMI-Traps species are missing in AMI-GBIF database.")

    with open(
        f"{ECCV2024_DATA}/camera_ready_amitraps/metadata/{missing_sp_fname}",
        "wb",
    ) as f:
        pickle.dump(missing_sp, f)


if __name__ == "__main__":
    gbif_taxonomy_map = pd.read_csv(
        f"{ECCV2024_DATA}/models/fine_grained/taxonomy_map_ami-gbif.csv"
    )
    insect_crops_dir = f"{ECCV2024_DATA}/camera_ready_amitraps/insect_crops"

    save_missing_species_keys(insect_crops_dir, gbif_taxonomy_map)
