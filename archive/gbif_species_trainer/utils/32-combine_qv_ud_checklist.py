"""
Author	           : Aditya Jain
Date last modified : October 17, 2023
About	           : Script to combine UK-Denmark and Quebec-Vermont checklist
"""

import pandas as pd

# Read checklists and defined the global list
qv_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/Quebec-Vermont_Moth-List_17Oct2023.csv"
)
ud_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_14Sep2023.csv"
)
global_checklist = ud_checklist
global_checklist_name = "Quebec-Vermont-UK-Denmark_Moth-List_17Oct2023.csv"
global_taxon_keys = global_checklist.loc[:, "accepted_taxon_key"].tolist()
write_dir = "/home/mila/a/aditya.jain/mothAI/species_lists/"
common_species = 0
source_tag = "quebec_vermont"

# Combine the two checklists with unique entries
for _, row in qv_checklist.iterrows():
    taxon_key = row["accepted_taxon_key"]
    species = row["gbif_species"]

    if taxon_key == -1 or taxon_key not in global_taxon_keys:
        global_checklist = global_checklist.append(row, ignore_index = True)
    else:
        idx = global_checklist["accepted_taxon_key"] == taxon_key
        global_checklist.loc[idx, "source"] = global_checklist.loc[idx, "source"] + " " + source_tag
        print(f"{species} exist in both checklists.")
        common_species += 1

# Save!
global_checklist.to_csv(write_dir + global_checklist_name, index=False)
print(f"Quebec-Vermont and UK-Denmark have {common_species} species in common.")
