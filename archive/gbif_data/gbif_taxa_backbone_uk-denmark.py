#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Author.      : Aditya Jain
Date Started : July 9, 2022
About        : This script fetches unique IDs for UK and Denmark moth species from GBIF database
               and builds a consolidated database
"""


# In[ ]:


from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import os
import tqdm
import urllib
import json
import time

data_dir              = '/home/mila/a/aditya.jain/mothAI/species_lists/'
uk_species_list       = 'uksi_moths_3-5-22.csv'
denmark_species_list  = 'Denmark_original_May22.csv'


# In[ ]:


def get_gbif_key_backbone(name, place):
    """ given a species name, this function returns the unique gbif key and other 
        attributes using backbone API
    """
    
    # default values
    taxon_key      = [-1]
    order          = ['NA']
    family         = ['NA']
    genus          = ['NA']    
    search_species = [name]
    gbif_species   = ['NA']     # the name returned on search, can be different from the search
    confidence     = ['']
    status         = ['NA']
    match_type     = ['NONE']
    place          = [place]

    data = species_api.name_backbone(name=name, strict=True, rank='species')

    if data['matchType'] == 'NONE':
        confidence    = [data['confidence']]
    else:
        taxon_key     = [data['usageKey']]
        order         = [data['order']]
        family        = [data['family']]
        genus         = [data['genus']]
        confidence    = [data['confidence']]
        gbif_species  = [data['species']]
        status        = [data['status']]
        match_type    = [data['matchType']]
  
    df = pd.DataFrame(list(zip(taxon_key, order, family, genus,
                               search_species, gbif_species, confidence,
                               status, match_type, place)),
                    columns =['taxon_key_gbif_id', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type', 'source'])
    return df


# ### Finding keys for UK moth list
# 

# In[ ]:


file             = data_dir + uk_species_list
uk_data          = pd.read_csv(file, index_col=False)
uk_species       = []

for indx in uk_data.index:
    if uk_data['taxon_rank'][indx]=='Species' and uk_data['preferred'][indx]==True:
        uk_species.append(uk_data['preferred_taxon'][indx])


# In[ ]:


data_final = pd.DataFrame(columns =['taxon_key_gbif_id', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type', 'source'], dtype=object)
for name in uk_species:
    data       = get_gbif_key_backbone(name, 'uksi_09May2022')
    data_final = data_final.append(data, ignore_index = True)
    
data_final.to_csv(data_dir + 'UK-Moth-List_11July2022.csv', index=False)  


# #### Counting the number of not-found entries

# In[ ]:


count = 0

for indx in data_final.index:
    if data_final['taxon_key_gbif_id'][indx] == -1:
        count += 1

print(f'The count of not found species for UK: {count}')        


# ### Finding keys for Denmark moth list

file             = data_dir + denmark_species_list
denmark_data     = pd.read_csv(file, index_col=False)
denmark_species  = []

for indx in denmark_data.index:
    denmark_species.append(denmark_data['species_name'][indx])


# In[ ]:


data_final = pd.DataFrame(columns =['taxon_key_gbif_id', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type', 'source'], dtype=object)
for name in denmark_species:
    data       = get_gbif_key_backbone(name, 'denmark_May2022')
    data_final = data_final.append(data, ignore_index = True)
    
data_final.to_csv(data_dir + 'Denmark-Moth-List_11July2022.csv', index=False)  


# #### Counting the number of not-found entries

# In[ ]:


count = 0

for indx in data_final.index:
    if data_final['taxon_key_gbif_id'][indx] == -1:
        count += 1

print(f'The count of not found species for Denmark: {count}')        

