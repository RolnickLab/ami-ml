#!/usr/bin/env python
# coding: utf-8

"""
Author       : Aditya Jain
Date Started : May 10, 2022
About        : This script scraps data from GBIF for uk list of moth species
"""

import pygbif
from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import os
import tqdm
import urllib
import json
import time
import math
import plotly.express as px

data_dir       = '/home/mila/a/aditya.jain/scratch/GBIF_Data/'
write_dir      = data_dir + 'moths_uk/'
INat_KEY       = '50c9509d-22c7-4a22-a47d-8c48425ef4a7'   # iNat key to fetch data from GBIF
LIMIT_DOWN     = 300                                      # GBIF API parameter for max results per page
MAX_DATA_SP    = 1000                                     # max. no of images to download for a species
MAX_SEARCHES   = 11000                                    # maximum no. of points to iterate

moth_species   = 'UK-MacroMoth-List_09May2022.csv'
file           = data_dir + moth_species
moth_data      = pd.read_csv(file)


# Downloading gbif data

def inat_metadata_gbif(data):
    """ returns the relevant gbif metadata for an iNat observation"""

    fields    = ['decimalLatitude', 'decimalLongitude',
            'order', 'family', 'genus', 'species', 'acceptedScientificName',
            'year', 'month', 'day',
            'datasetName', 'taxonID', 'acceptedTaxonKey', 'lifeStage', 'datasetName']

    meta_data = {}

    for field in fields:
        if field in data.keys():
            meta_data[field] = data[field]
        else:
            meta_data[field] = ''

    return meta_data


dataset_list       = []

taxon_key          = list(moth_data['taxon_key_guid'])         # list of taxon keys
species_name       = list(moth_data['search_species_name'])    # list of species name that is searched
gbif_species_name  = list(moth_data['gbif_species_name'])      # list of species name returned by gbif [can be different from above or NA]
columns            = ['taxon_key_guid', 'search_species_name', 'gbif_species_name', 'count']
count_list         = pd.DataFrame(columns = columns, dtype=object)         # uncomment if downloading data from scratch       

### this snippet is run ONLY is training is resuming from some point ####
start              = 858
end                = ''
taxon_key          = taxon_key[start:]
species_name       = species_name[start:]
gbif_species_name  = gbif_species_name[start:]
count_list         = pd.read_csv(write_dir + 'datacount.csv')         # keeps the count of data downloaded for each species: key, name, name, count                            
##########################################################################

for i in range(len(taxon_key)):
    print('Downloading for: ', species_name[i])
    begin   = time.time()
    if taxon_key[i] == -1:          # taxa not there on GBIF
        count_list = count_list.append(pd.DataFrame([['NA', species_name[i], 'NA', 'NA']],
                                                columns=columns), ignore_index=True)
    else: 
        data        = occ.search(taxonKey = int(taxon_key[i]), mediatype='StillImage', limit=1)
        total_count = data['count']   

        if total_count==0:            # no image data found for the species 
            count_list = count_list.append(pd.DataFrame([[taxon_key[i], species_name[i], gbif_species_name[i], 0]],
                                                columns=columns), ignore_index=True)    
        else:
            image_count = 0                                   # images downloaded for every species
            max_count   = min(total_count, MAX_DATA_SP)
            total_pag   = math.ceil(MAX_SEARCHES/LIMIT_DOWN)  # total pages to be fetched with max 300 entries each
            offset      = 0

            family  = data['results'][0]['family']
            genus   = data['results'][0]['genus']  
            species = data['results'][0]['species']

            m_data  = {}                                 # dictionary variable to store metadata
            write_loc = write_dir + family + "/" + genus + "/" + species 

            try:    
                os.makedirs(write_loc)                     # creating hierarchical structure for image storage 
            except:
                pass

            for j in range(total_pag):
                try:
                    data       = occ.search(taxonKey = int(taxon_key[i]), mediatype='StillImage', 
                               limit=LIMIT_DOWN, offset=offset)        
                except:
                    tot_points = 0
                else:
                    tot_points = len(data['results'])

                for k in range(tot_points):                     
                    if data['results'][k]['media'] and 'lifeStage' in data['results'][k].keys():
                        if data['results'][k]['lifeStage']=='Adult':            
                            gbifid   = data['results'][k]['gbifID']
                            
                            if 'identifier' in data['results'][k]['media'][0].keys():
                                image_url   = data['results'][k]['media'][0]['identifier']            
                                try:
                                    urllib.request.urlretrieve(image_url, write_loc + '/' + gbifid + '.jpg')
                                    image_count += 1              
                                    meta_data      = inat_metadata_gbif(data['results'][k])   # fetching metadata
                                    m_data[gbifid] = meta_data                                
                                    if meta_data['datasetName'] not in dataset_list:
                                        dataset_list.append(meta_data['datasetName'])
                                except:
                                    pass  
                    if image_count >= MAX_DATA_SP:
                        break
                
                offset += LIMIT_DOWN
                if image_count >= MAX_DATA_SP:
                    break

            with open(write_loc + '/' + 'metadata.txt', 'w') as outfile:
                json.dump(m_data, outfile)       

            count_list = count_list.append(pd.DataFrame([[int(taxon_key[i]), species_name[i], 
                         gbif_species_name[i], image_count]],
                                                columns=columns), ignore_index=True)
      
            end = time.time()
            print('Time taken to download data for ', gbif_species_name[i], ' is - ', 
            round(end-begin), 'sec for ', image_count, ' images')

    count_list.to_csv(write_dir + 'datacount.csv', index=False)
      
print(count_list)    
print(dataset_list)




