#!/bin/bash
#SBATCH --job-name=binary_model_evaluation_%j
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --output=binary_model_evaluation_%j.out

## Run this from the projet root directory.
## ./research/<sub_dir(s)>/<filename>.sh

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

# 4. Run python file
python binary_model_evaluation.py \
--run_name binary_resnet50_baseline_run3 \
--wandb_model_artifact moth-ai/ami-gbif-binary/model:v1 \
--skip_small_crops True \
--min_crop_dim 150 \
--model_type resnet50 \
--model_dir $ECCV2024_DATA/models/binary \
--category_map 05-moth-nonmoth_category_map.json \
--insect_crops_dir $ECCV2024_DATA/camera_ready_amitraps/insect_crops \
