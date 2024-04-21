#!/bin/bash
#SBATCH --job-name=fgrained_model_evaluation1
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --output=fgrained_model_evaluation1.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Launch your script(s)
python fgrained_model_evaluation.py \
--run_name america_resnet50_baseline_run1 \
--wandb_model_artifact moth-ai/ami-gbif-fine-grained/model:v13 \
--region NorthEasternAmerica \
--model_type resnet50 \
--category_map 01_ami-gbif_fine-grained_ne-america_category_map.json \

--sp_exclusion_list_file /home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle \
--insect_crops_dir /home/mila/a/aditya.jain/scratch/eccv2024_data/camera_ready_amitraps/insect_crops \
--model_dir /home/mila/a/aditya.jain/scratch/eccv2024_data/models/fine_grained \







