#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for main job
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --mem=10G                             # Ask for 10 GB of RAM 

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. run the python file
python tracks_w_classification-multiple.py 
