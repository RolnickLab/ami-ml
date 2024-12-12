#!/bin/bash
#SBATCH --partition=main                # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=4G                              # Ask for 4 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 4. Launch your script
# python create_webdataset.py \
# --dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk/ \
# --dataset_filepath /home/mila/a/aditya.jain/mothAI/classification_moths/data/01-uk-train-split.csv \
# --label_filepath /home/mila/a/aditya.jain/mothAI/classification_moths/data/uk_numeric_labels.json \
# --image_resize 500 \
# --webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/train/train-500-%06d.tar" \
# --max_shard_size 100*1024*1024

python create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk-denmark/ \
--dataset_filepath /home/mila/a/aditya.jain/mothAI/classification_moths/data/01-uk-val-split.csv \
--label_filepath /home/mila/a/aditya.jain/mothAI/classification_moths/data/uk_numeric_labels.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/val/val-500-%06d.tar" \
--max_shard_size 100*1024*1024

# python create_webdataset.py \
# --dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk/ \
# --dataset_filepath /home/mila/a/aditya.jain/mothAI/classification_moths/data/01-uk-test-split.csv \
# --label_filepath /home/mila/a/aditya.jain/mothAI/classification_moths/data/uk_numeric_labels.json \
# --image_resize 500 \
# --webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/test/test-500-%06d.tar" \
# --max_shard_size 100*1024*1024