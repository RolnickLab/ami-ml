#!/bin/bash
#SBATCH --partition=long-cpu                 # Ask for long-cpu job
#SBATCH --cpus-per-task=4                    # Ask for 2 CPUs
#SBATCH --mem=4G                             # Ask for 4 GB of RAM


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 4. Launch your script
python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--dataset_filepath /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/01-uk-denmark_train-split.csv \
--label_filepath /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels_25Apr2023.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/train/train-500-%06d.tar" 

python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--dataset_filepath /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/01-uk-denmark_val-split.csv \
--label_filepath /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels_25Apr2023.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/val/val-500-%06d.tar" 


python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--dataset_filepath /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/01-uk-denmark_test-split.csv \
--label_filepath /home/mila/a/aditya.jain/mothAI/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels_25Apr2023.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/test/test-500-%06d.tar" 
