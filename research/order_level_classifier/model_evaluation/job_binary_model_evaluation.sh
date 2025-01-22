#!/bin/bash
#SBATCH --job-name=order_model_evaluation_%j
#SBATCH --ntasks=1
#SBATCH --time=1:30:00
#SBATCH --partition=long              # Ask for long job
#SBATCH --cpus-per-task=2             # Ask for 2 CPUs
#SBATCH --gres=gpu:1                  # Ask for 1 GPU
#SBATCH --mem=10G                     # Ask for 10 GB of RAM
#SBATCH --output=order_model_evaluation_%j.out

## Run this from the projet root directory.
## ./research/<sub_dir(s)>/<filename>.sh

# 1. Load the required modules
module load miniconda/3

# 2. Load your environment
conda activate ami-ml

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

# 4. Run python file
python order_level_classifier/model_evaluation/binary_model_evaluation.py
