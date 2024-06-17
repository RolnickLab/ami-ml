#!/bin/bash
#SBATCH --partition=unkillable-cpu          # Ask for unkillable job
#SBATCH --cpus-per-task=2                   # Ask for 2 CPUs
#SBATCH --mem=4G                            # Ask for 4 GB of RAM
#SBATCH --time=6:00:00

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

# 4. Download data
aws s3 sync $ECCV2024_DATA_OBJECT_STORE $TEST_PATH
