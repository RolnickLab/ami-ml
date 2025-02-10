#!/bin/bash
#SBATCH --partition=long-cpu                  # Ask for long-cpu job
#SBATCH --mem=4G                              # Ask for 4 GB of RAM


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 4. Launch your script
python 02-calculate_taxa_statistics.py \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_25Apr2023.csv \
--write_dir /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/ \
--numeric_labels_filename uk-denmark_numeric_labels_25Apr2023 \
--category_map_filename uk-denmark_category_map_25Apr2023 \
--taxon_hierarchy_filename uk-denmark_taxon_hierarchy_25Apr2023 \
--training_points_filename uk-denmark_count_training_points_25Apr2023 \
--train_split_file /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/01-uk-denmark_train-split.csv


