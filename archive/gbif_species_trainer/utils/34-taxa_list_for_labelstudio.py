"""
Author	           : Aditya Jain
Date last modified : September 14, 2023
About	           : Script to make a json list of species to be imported in Label Studio for the AMI test set preparation
"""

import pandas as pd
import json

# Read the master checklist
master_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/Quebec-Vermont-UK-Denmark_Moth-List_14Sep2023.csv"
)
final_json = []
count = 0

# Prepare the json file
for _, row in master_checklist.iterrows():
    search_species = row["search_species"]
    gbif_species = row["gbif_species"]
    genus = row["genus"]
    family = row["family"]
    gbif_taxon_key = row["accepted_taxon_key"]
    status = row["status"]

    if gbif_taxon_key == -1:
        final_json.append(
            {
                "species": search_species,
                "genus": genus,
                "family": family,
                "gbif_taxon_key": gbif_taxon_key,
            }
        )
    elif status == "SYNONYM":
        final_json.append(
            {
                "species": search_species,
                "genus": genus,
                "family": family,
                "gbif_taxon_key": gbif_taxon_key,
                "synonym_of": gbif_species,
            }
        )
        final_json.append(
            {
                "species": gbif_species,
                "genus": genus,
                "family": family,
                "gbif_taxon_key": gbif_taxon_key,
            }
        )
        count += 1
    else:
        final_json.append(
            {
                "species": gbif_species,
                "genus": genus,
                "family": family,
                "gbif_taxon_key": gbif_taxon_key,
            }
        )

    count += 1

# Write json file
with open("quebec-vermont-uk-denmark-taxa-20230914.json", "w") as outfile:
    json.dump(final_json, outfile, indent=4)

print(f"Total entries are {count}.")
