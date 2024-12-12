#!/bin/bash
#SBATCH --partition=long                # Ask for main job
#SBATCH --cpus-per-task=2                     # Ask for 1 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=5G                             # Ask for 10 GB of RAM

## this bash script archives the moth data into one file

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

start=`date +%s`

python find_weird_images.py

end=`date +%s`

runtime=$((end-start))
echo 'Time taken for finding the data in seconds' $runtime

