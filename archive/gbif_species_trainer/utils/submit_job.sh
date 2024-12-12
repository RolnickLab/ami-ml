#!/bin/bash
#SBATCH --partition=long                # Ask for long job
#SBATCH --cpus-per-task=2               # Ask for 1 CPUs
#SBATCH --mem=5G                        # Ask for 10 GB of RAM

## this bash script archives the moth data into one file

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

start=`date +%s`

python copy_gbif_data_to_root_folder.py

end=`date +%s`

runtime=$((end-start))
echo 'Time taken for running the script in seconds' $runtime

