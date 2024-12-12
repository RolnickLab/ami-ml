"""
Author	           : Aditya Jain
Date last modified : August 24, 2023
About	           : Script to combine UK and Denmark checklist
"""

import pandas as pd
import os


# Read checklists and defined the global list
uk_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/UK_Moth-List_25Apr2023.csv"
)
denmark_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/Denmark_Moth-List_25Apr2023.csv"
)
global_checklist = denmark_checklist
global_checklist_name = "UK-Denmark_test_checklist.csv"
global_taxon_keys = global_checklist.loc[:, "accepted_taxon_key"].tolist()
write_dir = "/home/mila/a/aditya.jain/mothAI/species_lists/"
common_species = 0

# Combine the two checklists with unique entries
for _, row in uk_checklist.iterrows():
    taxon_key = row["accepted_taxon_key"]
    species = row["gbif_species_name"]

    if taxon_key == -1 or taxon_key not in global_taxon_keys:
        global_checklist = global_checklist.append(row, ignore_index = True)
    else:
        print(f"{species} exist in both checklists.")
        common_species += 1

# Save!
global_checklist.to_csv(write_dir + global_checklist_name, index=False)
print(f"UK and Denmark have {common_species} species in common.")
