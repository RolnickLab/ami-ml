'''
Author.      : Aditya Jain
Date Started : 30th June, 2021
About        : This script fetches data for non-moth taxas
'''

from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import os
import tqdm
import urllib
import json
import time
import math  

DATA_DIR       = '/miniscratch/transit_datasets/restricted/inat_users/GBIF_Data/'
WRITE_DIR      = DATA_DIR + 'nonmoths/'
LIMIT_DOWN     = 300                                      # GBIF API parameter for max results per page
MAX_DATA_SP    = 30000                                    # max. no of images to download for a taxa
MAX_SEARCHES   = 45000                                    # maximum no. of points to iterate

nonmoth_taxa   = 'NonMothList_30June2021.csv'
file           = DATA_DIR + nonmoth_taxa
nonmoth_data   = pd.read_csv(file)

def inat_metadata_gbif(data):
    '''
    this function returns the relevant gbif metadata for an iNat observation
    '''
    fields    = ['decimalLatitude', 'decimalLongitude', 'phylum',
            'order', 'family', 'genus', 'species', 'acceptedScientificName',
            'year', 'month', 'day', 'datasetName', 'taxonID', 'acceptedTaxonKey']

    meta_data = {}

    for field in fields:
        if field in data.keys():
            meta_data[field] = data[field]
        else:
            meta_data[field] = ''

    return meta_data


taxon_key          = list(nonmoth_data['Taxon Key'])        # list of taxon keys
taxon_name         = list(nonmoth_data['Search Name'])      # list of species name that is searched
gbif_taxon_name    = list(nonmoth_data['GBIF Taxa Name'])   # list of species name returned by gbif
count_list         = list(nonmoth_data['Count Download'])

### this snippet is run ONLY is training is resuming from some point ####
start              = 6
end                = ''
taxon_key          = taxon_key[start:]
taxon_name         = taxon_name[start:]
gbif_taxon_name    = gbif_taxon_name[start:]
count_list         = count_list[start:]
##########################################################################

for i in range(len(taxon_key)):
    print('Downloading for: ', gbif_taxon_name[i])
    begin       = time.time()
    data        = occ.search(taxonKey = taxon_key[i], mediatype='StillImage', limit=1)
    total_count = data['count'] 
    
    MAX_DATA_SP = count_list[i]
    print('Maximum images to download: ', MAX_DATA_SP)
    
    if total_count==0:            # no data for the species on iNat
        print('No image record!')   
    else:
        image_count = 0                                   # images downloaded for every species
        max_count   = min(total_count, MAX_DATA_SP)
        total_pag   = math.ceil(MAX_SEARCHES/LIMIT_DOWN)  # total pages to be fetched with max 300 entries each
        offset      = 0  
        m_data      = {}                                 # dictionary variable to store metadata
        
        write_loc   = WRITE_DIR + gbif_taxon_name[i] 
        try:    
            os.makedirs(write_loc)                     # creating hierarchical structure for image storage 
        except:
            pass
        
        for j in range(total_pag):
            data       = occ.search(taxonKey = taxon_key[i], mediatype='StillImage', 
                               limit=LIMIT_DOWN, offset=offset)
            tot_points = len(data['results'])
            
            for k in range(tot_points):                
                if data['results'][k]['media']: 
                    gbifid   = data['results'][k]['gbifID']
                
                    if 'identifier' in data['results'][k]['media'][0].keys():
                        image_url   = data['results'][k]['media'][0]['identifier']            
                        try:
                            urllib.request.urlretrieve(image_url, write_loc + '/' + gbifid + '.jpg')
                            image_count += 1              
                            meta_data      = inat_metadata_gbif(data['results'][k])   # fetching metadata
                            m_data[gbifid] = meta_data                 
                        except:
                            pass
                        
                if image_count >= max_count:
                    break
                    
            with open(write_loc + '/' + 'metadata.txt', 'w') as outfile:
                json.dump(m_data, outfile)
                
            offset += LIMIT_DOWN
            if image_count >= max_count:
                break
                
        
                
        end = time.time()
        print('Time taken to download data for ', gbif_taxon_name[i], ' is - ',round(end-begin), 'sec for ', image_count, ' images')
