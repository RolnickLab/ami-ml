"""
Author	           : Aditya Jain
Date last modified : November 24, 2023
About	           : Combine two checklist based on taxon keys
"""

import pandas as pd

# Read checklists and defined the global list
main_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/Quebec-Vermont-UK-Denmark_Moth-List_17Oct2023.csv"
)
incoming_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/panama-bci_checklist_20231124.csv"
)
combined_checklist = main_checklist
combined_checklist_name = "quebec-vermont-uk-denmark-panama_checklist_20231124.csv"
combined_taxon_keys = combined_checklist.loc[:, "accepted_taxon_key"].tolist()
write_dir = "/home/mila/a/aditya.jain/mothAI/species_lists/"
common_species = 0
incoming_list_tag = "panama-bci"

# Combine the two checklists with unique entries
for _, row in incoming_checklist.iterrows():
    taxon_key = row["accepted_taxon_key"]
    species = row["gbif_species"]

    if taxon_key == -1 or taxon_key not in combined_taxon_keys:
        combined_checklist = combined_checklist.append(row, ignore_index = True)
    else:
        idx = combined_checklist["accepted_taxon_key"] == taxon_key
        combined_checklist.loc[idx, "source"] = combined_checklist.loc[idx, "source"] + " " + incoming_list_tag
        print(f"{species} exist in both checklists.")
        common_species += 1

# Save!
combined_checklist.to_csv(write_dir + combined_checklist_name, index=False)
print(f"The two checklists have {common_species} species in common.")
