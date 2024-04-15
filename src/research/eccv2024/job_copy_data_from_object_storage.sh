#!/bin/bash
#SBATCH --partition=unkillable-cpu          # Ask for unkillable job
#SBATCH --cpus-per-task=2                   # Ask for 2 CPUs
#SBATCH --mem=4G                            # Ask for 4 GB of RAM
#SBATCH --time=6:00:00

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Download data
aws s3 sync s3://ami-trapdata/cvpr2024-test-set/ami_traps_dataset/labels /home/mila/a/aditya.jain/scratch/cvpr2024_data/ami_traps_dataset/labels