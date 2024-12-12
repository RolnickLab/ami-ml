#!/bin/bash
#SBATCH --partition=long-cpu                  # Ask for long-cpu job
#SBATCH --mem=4G                              # Ask for 4 GB of RAM


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 4. Launch your script
python 01-create_dataset_split.py \
--root_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--write_dir /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/ \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_25Apr2023.csv \
--train_ratio 0.75 \
--val_ratio 0.10 \
--test_ratio 0.15 \
--filename 01-uk-denmark


