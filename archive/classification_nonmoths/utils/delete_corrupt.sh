#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs

# deletes the corrupted images in the dataset 

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. run the python file
python 02-02-build_localiz_annot.py 
