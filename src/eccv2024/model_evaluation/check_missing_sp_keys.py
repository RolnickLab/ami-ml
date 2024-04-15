"""
Author: Aditya Jain
Date last modified: March 6th, 2023
About: Check for missing species keys in GBIF database
"""

import pandas as pd
import json
import pickle

sp_key_map_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/speciesKey_map.csv"
sp_key_map = pd.read_csv(sp_key_map_file)
ami_traps_sp_file = "/home/mila/a/aditya.jain/mothAI/eccv2024/model_evaluation/plots/hist_species.json"
ami_traps_sp = json.load(open(ami_traps_sp_file))
sp_list_file = "/home/mila/a/aditya.jain/mothAI/species_lists/quebec-vermont-uk-denmark-panama_checklist_20231124.csv"
sp_list = pd.read_csv(sp_list_file)
gbif_count_file = "/home/mila/a/aditya.jain/scratch/eccv2024_data/gbif_train_counts.json"
gbif_count = json.load(open(gbif_count_file))
exclusion_sp = []

for species in ami_traps_sp.keys():    
    gbif_name = sp_list.loc[sp_list["gbif_species"] == species]
    search_name = sp_list.loc[sp_list["search_species"] == species]
    gbif_test = search_name["gbif_species"]

    # Species not available completely
    if gbif_name.empty and search_name.empty:
        exclusion_sp.append(species)
        print(f"{species} is not available in our GBIF checklist.")

    # Search available but not found on GBIF
    elif gbif_name.empty:
        exclusion_sp.append(species)
        print(f"{species} is not found in the GBIF backbone.")

    else:
        numeric_id = gbif_name["accepted_taxon_key"].values[0]
        
        # Find image count using taxon key
        try:
            count = gbif_count[str(numeric_id)]  
        except:
            exclusion_sp.append(species)
            print(f"{species} with id {numeric_id} has no count in the data file.")


print(f"Total species in AMI-Traps absent in AMI-GBIF are {len(exclusion_sp)}.")
# with open("/home/mila/a/aditya.jain/scratch/eccv2024_data/excluded_sp_from_AMI-GBIF.pickle", "wb") as f:
#     pickle.dump(exclusion_sp, f)