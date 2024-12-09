#!/bin/bash
#SBATCH --job-name=fetch_gbif_images
#SBATCH --partition=unkillable-cpu          # Ask for unkillable-cpu job
#SBATCH --cpus-per-task=4                   # Ask for 4 CPUs
#SBATCH --mem=300G                          # Ask for 300 GB of RAM
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
--dataset-path $ORDER_CLASSIFIER_RAW_DATA \
--dwca-file $ORDER_CLASSIFIER_DWCA \
--num-images-per-category 25000 \
--num-workers 4 \
--subset-list $ORDER_CLASSIFIER_ACCEPTED_KEYS

# Print time taken to execute the script
echo "Time taken to fetch images: $SECONDS seconds"
