#!/bin/bash
#SBATCH --job-name=export_to_webdataset
#SBATCH --partition=main-cpu                 # Ask for main-cpu job
#SBATCH --cpus-per-task=1                    # Ask for 1 CPUs
#SBATCH --mem=5G                             # Ask for 5 GB of RAM
#SBATCH --output=export_to_webdataset_and_crops.out

## Run this from the projet root directory.
## ./research/<sub_dir(s)>/<filename>.sh

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate ami-ml

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

# 4. Launch your script
poetry run python eccv2024/export_to_webdataset_and_crops.py
