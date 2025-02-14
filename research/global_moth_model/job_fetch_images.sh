#!/bin/bash
#SBATCH --job-name=fetch_gbif_images
#SBATCH --partition=long-cpu                 # Ask for long-cpu job
#SBATCH --cpus-per-task=1                   # Ask for 1 CPUs
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

# Keep track of time
SECONDS=0

# 4. Launch your script
ami-dataset fetch-images \
--dataset-path $GLOBAL_MODEL_DATASET_PATH \
--dwca-file $DWCA_FILE \
--num-images-per-category 1000 \
--num-workers 4 \
--subset-list $ACCEPTED_KEY_LIST

# Print time taken to execute the script
echo "Time taken: $SECONDS seconds"
