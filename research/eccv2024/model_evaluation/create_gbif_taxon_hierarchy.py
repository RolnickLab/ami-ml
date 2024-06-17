"""
Author: Aditya Jain
Date started: April 21, 2024
About: Create taxon hierarchy for gbif taxon species
"""

import json
import os
from pathlib import Path

import pandas as pd


def save_taxonomy_map(data_dir: str, taxonomy_map: str):
    """Save taxonomy map"""

    # Variables
    gbif_taxonomy_map = pd.read_csv(Path(data_dir) / taxonomy_map)
    taxon_hierarchy = {}

    for _, row in gbif_taxonomy_map.iterrows():
        taxon_hierarchy[row["speciesKey"]] = [row["genus"], row["family"]]

    with open(Path(data_dir) / "gbif_taxonomy_hierarchy.json", "w") as f:
        json.dump(taxon_hierarchy, f)


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA")
    data_dir = f"{ECCV2024_DATA}/camera_ready_amitraps/metadata"
    taxonomy_map = "ami-gbif_taxonomy_map.csv"

    save_taxonomy_map(data_dir, taxonomy_map)
