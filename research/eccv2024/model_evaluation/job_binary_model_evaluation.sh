#!/bin/bash
#SBATCH --job-name=binary_model_evaluation5
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --output=binary_model_evaluation5.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Launch your script(s)
python binary_model_evaluation.py \
--run_name binary_resnet50_baseline_run2 \
--wandb_model_artifact moth-ai/ami-gbif-binary/model:v2 \
--skip_small_crops True \
--min_crop_dim 100 \
--model_type resnet50 \
--model_dir /home/mila/a/aditya.jain/scratch/eccv2024_data/models/binary \
--category_map 05-moth-nonmoth_category_map.json \
--insect_crops_dir /home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/insect_crops \



