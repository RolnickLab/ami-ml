#!/bin/bash
#SBATCH --partition=long-cpu          # Ask for unkillonglable job
#SBATCH --cpus-per-task=1             # Ask for 2 CPUs
#SBATCH --mem=4G                      # Ask for 4 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Launch your script
python 03-update_data_statistics.py \
--data_directory /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_25Apr2023.csv
