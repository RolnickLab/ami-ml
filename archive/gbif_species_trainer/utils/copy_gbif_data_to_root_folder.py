#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Author        : Aditya Jain
Date Started  : April 10th, 2023
About         : Copies GBIF data from region specifc folder to the world data folder
"""
import os
import glob
import pandas as pd
import shutil
from pygbif import species as species_api

source_dir    = '/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_panama/'    
target_dir    = '/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/'


# Read the species data already present in the moths world folder
columns = [
        "taxon_key_gbif_id",
        "family_name",
        "genus_name",
        "search_species_name",
        "gbif_species_name",
        "image_count",
        "total_occ_count",
    ]

data_type = {
        "taxon_key_gbif_id": int,
        "family_name": str,
        "genus_name": str,
        "search_species_name": str,
        "gbif_species_name": str,
        "image_count": int,
        "total_occ_count": int,
    }

df_world_data = pd.read_csv(target_dir + 'data_statistics.csv', dtype=data_type)
world_species = list(df_world_data['gbif_species_name'])


# Copy the data to the target directory
# traverse over family names
for family in os.listdir(source_dir):
    if os.path.isdir(source_dir + '/' + family):
        
        # traverse over genus names
        for genus in os.listdir(source_dir + family):
            if os.path.isdir(source_dir + '/' + family + '/' + genus):
                
                # traverse over species names
                for species in os.listdir(source_dir + family + '/' + genus):
                    if os.path.isdir(source_dir + '/' + family + '/' + genus + '/' + species): 
                        
                        image_list  = glob.glob(source_dir + family + '/' + genus + '/' + species + '/*.jpg')
                        if species not in world_species and len(image_list)!=0:
                            source_folder = source_dir + '/' + family + '/' + genus + '/' + species
                            target_folder = target_dir + '/' + family + '/' + genus + '/' + species
                            
                            # delete folder and its content, if exists already
                            try:
                                shutil.rmtree(target_folder)
                                print(f'{target_folder} already existed but now deleted.', flush=True)
                            except:
                                pass
                              
                            # copy the data    
                            try:
                                shutil.copytree(source_folder, target_folder)
                                backbone_data = species_api.name_backbone(name=species, strict=True, rank='species')
                                df_world_data = pd.concat([df_world_data, pd.DataFrame([[backbone_data['usageKey'], family, genus, species, species, len(image_list), -1]], columns=columns)], ignore_index=True)
                                df_world_data.to_csv(target_dir + "data_statistics.csv", index=False)
                                print(f'Data copied for {species} with {len(image_list)} images.', flush=True)
                            except Exception as e:
                                print(e, flush=True)
                                print(f'Data already available in the target directory for {species}', flush=True)
                                continue                    


# In[ ]:




