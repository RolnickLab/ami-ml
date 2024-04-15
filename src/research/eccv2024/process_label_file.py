"""
Author: Aditya Jain
Date last modified: November 6, 2023
About: Add rank and hierarchy information to the label file
"""

import json
import os
import pandas as pd

# User-input variables
root_data_dir = "/home/mila/a/aditya.jain/scratch/cvpr2024_data"
yolo_data_dir = "ami-traps-dataset"
taxon_db_file = "ami-taxa-20231029.csv"


def process_label_file(data_dir: str, yolo_dir: str, taxon_db_file: str):
    """Main function to add rank and hierarchy information to the label file"""

    # Read required files and data
    orig_label_list = json.load(open(os.path.join(data_dir, "notes-original.json")))["categories"]
    new_label_list = {"categories": []}
    taxon_db = pd.read_csv(os.path.join(data_dir, taxon_db_file))

    # Iterate over each class entry
    for class_entry in orig_label_list:
        id = class_entry["id"]
        name = class_entry["name"]

        # Search for the rank in the database
        try:
            rank = taxon_db.loc[taxon_db["name"] == name, "rank"].values[0]
        except:
            print(f"{name} is not present in the database.")
            rank = "NA"

        new_entry = {"id": id, "name": name, "rank": rank}
        new_label_list["categories"].append(new_entry)
    
    # Write the file to disk
    with open(os.path.join(data_dir, yolo_dir, "notes.json"), "w") as f:
        json.dump(new_label_list, f, indent=4)


if __name__ == "__main__":
    process_label_file(root_data_dir, yolo_data_dir, taxon_db_file)
