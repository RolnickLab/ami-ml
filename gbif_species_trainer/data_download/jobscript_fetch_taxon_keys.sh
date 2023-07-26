#!/bin/bash
#SBATCH --partition=long-cpu                # Ask for long job
#SBATCH --cpus-per-task=1                     # Ask for 2 CPUs
#SBATCH --mem=4G                              # Ask for 4 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Launch your script
python 01-fetch_taxon_keys.py \
--species_filepath /home/mila/a/aditya.jain/mothAI/species_lists/Quebec-Vermont_Moth-List_22July2022.csv \
--column_name search_species_name \
--output_filepath /home/mila/a/aditya.jain/mothAI/species_lists/Quebec-Vermont_Moth-List_26July2023.csv \
--place quebec_vermont