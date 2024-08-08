#!/bin/bash
#SBATCH --job-name=fetch_gbif_images
#SBATCH --partition=long-cpu                 # Ask for long-cpu job
#SBATCH --cpus-per-task=64                   # Ask for 64 CPUs
#SBATCH --mem=300G                           # Ask for 300 GB of RAM
#SBATCH --output=fetch_gbif_images_%j.out

# 1. Load the required modules
module load miniconda/3

# 2. Load your environment
conda activate ami-ml

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

SECONDS=0
# 4. Launch your script
ami-dataset fetch-images \
--dataset-path $GLOBAL_MODEL_DATASET_PATH \
--dwca-file $DWCA_FILE \
--num-images-per-category 1000 \
--num-workers 32 \
--subset-list $ACCEPTED_KEY_LIST

echo "Time taken: $SECONDS seconds"
