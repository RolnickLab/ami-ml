"""
Author: Aditya Jain
Date last modified: November 6, 2023
About: Add rank and hierarchy information to the label file
"""

import json
import os

import pandas as pd

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()


def process_label_file(data_dir: str, yolo_dir: str, taxon_db_file: str):
    """Main function to add rank and hierarchy information to the label file"""

    # Read required files and data
    orig_label_list = json.load(open(os.path.join(data_dir, "notes-original.json")))[
        "categories"
    ]
    new_label_list = {"categories": []}
    taxon_db = pd.read_csv(os.path.join(data_dir, taxon_db_file))

    # Iterate over each class entry
    for class_entry in orig_label_list:
        id = class_entry["id"]
        name = class_entry["name"]

        # Search for the rank in the database
        try:
            rank = taxon_db.loc[taxon_db["name"] == name, "rank"].values[0]
        except IndexError:
            print(f"{name} is not present in the database.")
            rank = "NA"

        new_entry = {"id": id, "name": name, "rank": rank}
        new_label_list["categories"].append(new_entry)

    # Write the file to disk
    with open(os.path.join(data_dir, yolo_dir, "notes.json"), "w") as f:
        json.dump(new_label_list, f, indent=4)


if __name__ == "__main__":
    ECCV2024_DATA = os.getenv("ECCV2024_DATA_PATH")

    # User-input variables
    root_data_dir = ECCV2024_DATA
    yolo_data_dir = "ami-traps-dataset"
    taxon_db_file = "ami-taxa-20231029.csv"

    process_label_file(root_data_dir, yolo_data_dir, taxon_db_file)
