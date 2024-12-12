#!/bin/bash
#SBATCH --partition=main                      # Ask for main job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=4G                              # Ask for 4 GB of RAM


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 4. Launch your script
python 01-create_dataset_split.py \
--data_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_quebec-vermont/ \
--write_dir /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/ \
--train_ratio 0.75 \
--val_ratio 0.10 \
--test_ratio 0.15 \
--filename 01-quebec-vermont


