## Downloading Data from GBIF 

This folder contains code to download image data and associated metadata from GBIF. The user is **required to have a list of species names**.


The following steps need to be executed in order:

### Step 1: Fetch unique keys
Step 1 involves fetching unique accepted taxon keys for each species from GBIF taxonomy backbone. 
```bash
python 01-fetch_taxon_keys.py \
--species_filepath example_species_lists/example2.csv \
--column_name beetles_species_names \
--output_filepath beetles_30June2022.csv \ 
--place panama_11Jan2022
```
The description of the arguments to the script:
* `--species_filepath`: Path to the csv file containing list of species names. Example species lists are provided in the `example_species_lists` folder. **Required**.
* `--column_name`: Column name in the above csv file containing species' names. **Required**.
* `--output_filepath`: Output file path with csv as extension. **Required**.
* `--place`: A placeholder name which identifies the source of the species list - important when combining multiple lists. **Required**.

### Step 2: Download data
Step 2 involves downloading data using a [Darwin Core Archive](https://ipt.gbif.org/manual/en/ipt/latest/dwca-guide) (DwC-A) file exported from GBIF.  

```bash
python 02-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--dwca_file /home/mila/a/aditya.jain/scratch/GBIF_Data/leps_images_adult-imago.zip
--species_checklist beetles_30June2022.csv \
--max_images_per_species 1000
```

The description of the arguments to the script:

* `--write_directory`: Path to the folder to download the data. **Required**.
* `--dwca_file`: Path to the DwC-A file. **Required**.
* `--species_checklist`: Path to the species list obtained from `01-fetch_taxon_keys.py`. **Required**.
* `--max_images_per_species`: Maximum number of images to download for any species. Optional. **Default** is **1000**.

### Step 3: Update data statistics 
The final step is to update the statistics (image count of each species) of the downloaded data in the root folder.

```bash
python 03-update_data_statistics.py \
--data_directory /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_25Apr2023.csv
```

The description of the arguments to the script:

* `--data_directory`: Path to the folder where the image data is downloaded. **Required**.
* `--species_checklist`: Path to the species list obtained from `01-fetch_taxon_keys.py`. **Required**.


