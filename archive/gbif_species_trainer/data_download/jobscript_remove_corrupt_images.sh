#!/bin/bash
#SBATCH --partition=long-cpu                  # Ask for long-cpu job
#SBATCH --cpus-per-task=40                    # Ask for 40 CPUs
#SBATCH --mem=10G                             # Ask for 10 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Launch your script
python 03-remove_corrupt_images.py \
--data_directory /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_24Aug2023.csv \
--num_workers 40
