"""
Author	           : Aditya Jain
Date last modified : August 1, 2023
About	           : Script to delete data that does not belong to Quebec, Vermont, UK, or Denmark 
"""

import pandas as pd
import os
import shutil

# read checklists
qv_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/Quebec-Vermont_Moth-List_26July2023.csv"
)
ud_checklist = pd.read_csv(
    "/home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_25Apr2023.csv"
)

# aggregate family, genus, and species names
families_qv = set(qv_checklist.loc[:, "family_name"].tolist())
families_ud = set(ud_checklist.loc[:, "family_name"].tolist())
families_all = list(families_qv.union(families_ud))
genus_qv = set(qv_checklist.loc[:, "genus_name"].tolist())
genus_ud = set(ud_checklist.loc[:, "genus_name"].tolist())
genus_all = list(genus_qv.union(genus_ud))
species_qv = set(qv_checklist.loc[:, "gbif_species_name"].tolist())
species_ud = set(ud_checklist.loc[:, "gbif_species_name"].tolist())
species_all = list(species_qv.union(species_ud))

# delete data that does not belong to either of the two checklist
data_dir = "/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world"

for family in os.listdir(data_dir):
    if os.path.isdir(data_dir + "/" + family) and family not in families_all:
        shutil.rmtree(data_dir + "/" + family)
        print(f"Data for {family} family deleted.")
    else:
        if os.path.isdir(data_dir + "/" + family):
            for genus in os.listdir(data_dir + "/" + family):
                if genus not in genus_all:
                    shutil.rmtree(data_dir + "/" + family + "/" + genus)
                    print(f"Data for {genus} genus deleted.")
                else:
                    for species in os.listdir(data_dir + "/" + family + "/" + genus):
                        if species not in species_all:
                            shutil.rmtree(
                                data_dir + "/" + family + "/" + genus + "/" + species
                            )
                            print(f"Data for {species} species deleted.")
