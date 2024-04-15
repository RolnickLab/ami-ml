#!/bin/bash
#SBATCH --job-name=predictions_evaluation
#SBATCH --partition=main-cpu                 # Ask for main-cpu job
#SBATCH --cpus-per-task=1                    # Ask for 1 CPUs
#SBATCH --mem=5G                             # Ask for 5 GB of RAM
#SBATCH --output=predictions_evaluation.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 4. Launch your script
python predictions_evaluation.py



