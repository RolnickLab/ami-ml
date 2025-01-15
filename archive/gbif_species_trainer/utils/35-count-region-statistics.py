"""
Author	           : Aditya Jain
Date last modified : October 20, 2023
About	           : Script to count number of families, genuses, species, and images for a given checklist
"""

import pandas as pd
import json

# Read the species checklist and data statistics 
region = "UK-Denmark"
species_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_14Sep2023.csv"
)
data_statistics = pd.read_csv(
    "/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/data_statistics.csv"
)
img_count = 0
family_dict = {}
genus_dict = {}
species_dict = {}

# Prepare the json file
for _, row in species_checklist.iterrows():
    taxon_key = row["accepted_taxon_key"]
    family = row["family"]
    genus = row["genus"]
    species = row["gbif_species"]

    if taxon_key != -1:

        # Count total images
        idx = data_statistics.loc[data_statistics["accepted_taxon_key"] == taxon_key, "image_count"].values
        if idx.size > 0:
            img_count += idx[0]

        # Count families
        if family not in family_dict.keys():
            family_dict[family] = 1

        # Count genuses
        if genus not in genus_dict.keys():
            genus_dict[genus] = 1

        # Count species
        if species not in species_dict.keys():
            species_dict[species] = 1

        
print(f"Total number of families, genuses and species for {region} region are {len(family_dict.keys())}, {len(genus_dict.keys())} and {len(species_dict.keys())} respectively.")        
print(f"Total number of images for {region} region is {img_count}.")
