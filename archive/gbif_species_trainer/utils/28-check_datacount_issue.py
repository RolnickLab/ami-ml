#!/usr/bin/env python
# coding: utf-8

"""
Author	           : Aditya Jain
Date last modified : July 27, 2023
About	           : Script to check data count issues 
"""

import pandas as pd
import glob

data_dir = '/home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/'
data_statistics = data_dir + 'data_statistics.csv'
df = pd.read_csv(data_statistics, dtype=object)

for idx, _ in df.iterrows():    
    family = df.iloc[idx, 1] 
    genus = df.iloc[idx, 2]
    species = df.iloc[idx, 4]
    image_count_on_file = int(df.iloc[idx, -1])

    if species != 'NotAvail':
        image_count_on_disk = len(glob.glob(data_dir + family + '/' + genus + '/' + species + '/*.jpg'))
        
        if image_count_on_file != image_count_on_disk:
            print(f'{species}: File count is {image_count_on_file}; Disk count is {image_count_on_disk}')

## Following have mismatch in the datacounts in the first run
# Korscheltellus lupulina: File count is 1000; Disk count is 0
# Macrosaccus robiniella: File count is 91; Disk count is 0
# Elachista albidella: File count is 16; Disk count is 6
# Chrysoclista linneela: File count is 158; Disk count is 0
# Sparganothis unifasciana: File count is 354; Disk count is 3
# Epinotia nisella: File count is 775; Disk count is 94
# Epinotia cinereana: File count is 94; Disk count is 0
# Diaphania hyalinata: File count is 497; Disk count is 0
# Orthonama obstipata: File count is 1000; Disk count is 0
# Macaria bicolorata: File count is 126; Disk count is 0
# Furcula furcula: File count is 257; Disk count is 1000
# Clepsis coriacana: File count is 29; Disk count is 0
# Eriopygodes imbecillus: File count is 565; Disk count is 0
# Lamprotes caureum: File count is 214; Disk count is 0