#!/usr/bin/env python
# coding: utf-8

"""Create a mapping from taxon keys to species names"""

import json
import os
from pathlib import Path

# System packages
import pandas as pd

# 3rd party packages
from dotenv import load_dotenv

# Load secrets and config from optional .env file
load_dotenv()

# Variable definitions
GLOBAL_MODEL_DIR = os.getenv("GLOBAL_MODEL_DIR")
moth_list = pd.read_csv(Path(GLOBAL_MODEL_DIR) / "gbif_moth_checklist_07242024.csv")
map_dict = {}
map_file = Path(GLOBAL_MODEL_DIR) / "categ_to_name_map.json"

# Build the dict
for _, row in moth_list.iterrows():
    map_dict[int(row["acceptedTaxonKey"])] = row["species"]

# Save the dict
with open(map_file, "w") as file:
    json.dump(map_dict, file, indent=2)
