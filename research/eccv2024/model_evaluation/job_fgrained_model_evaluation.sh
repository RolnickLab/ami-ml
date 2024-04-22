#!/bin/bash
#SBATCH --job-name=fgrained_model_evaluation
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --output=fgrained_model_evaluation2.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Launch your script(s)
python fgrained_model_evaluation.py \
w-europe_vitb128_baseline_run2 \
moth-ai/ami-gbif-fine-grained/model:v54 \
WesternEurope \
timm_vit-b16-128 \
/home/mila/a/aditya.jain/scratch/eccv2024_data/models/fine_grained \
02_ami-gbif_fine-grained_w-europe_category_map.json \
/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/insect_crops \
/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle \
/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-traps_taxonomy_map.csv \
/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-gbif_taxonomy_map.csv \
/home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/gbif_taxonomy_hierarchy.json