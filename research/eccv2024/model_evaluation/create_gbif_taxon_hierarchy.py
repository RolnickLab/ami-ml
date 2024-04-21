"""
Author: Aditya Jain
Date started: April 21, 2024
About: Create taxon hierarchy for gbif taxon species
"""

import pandas as pd
import json
import os


def save_taxonomy_map(data_dir: str, taxonomy_map: str):
    """Save taxonomy map"""

    # Variables
    gbif_taxonomy_map = pd.read_csv(os.path.join(data_dir, taxonomy_map))
    taxon_hierarchy = {}

    for _, row in gbif_taxonomy_map.iterrows():
        taxon_hierarchy[row["speciesKey"]] = [row["genus"], row["family"]]

    with open(os.path.join(data_dir, "gbif_taxonomy_hierarchy.json"), "w") as f:
        json.dump(taxon_hierarchy, f)
        

if __name__ == "__main__":
    data_dir = "/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata"
    taxonomy_map = "ami-gbif_taxonomy_map.csv"

    save_taxonomy_map(data_dir, taxonomy_map)