#!/bin/bash
#SBATCH --job-name=verify_gbif_images
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu                 # Ask for long-cpu job
#SBATCH --cpus-per-task=16                   # Ask for 16 CPUs
#SBATCH --mem=300G                           # Ask for 300 GB of RAM
#SBATCH --output=verify_gbif_images_%j.out

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
ami-dataset verify-images \
--dataset-path $ORDER_CLASSIFIER_RAW_DATA \
--dwca-file $ORDER_CLASSIFIER_DWCA \
--num-workers 16 \
--results-csv $VERIFICATION_RESULTS_ORDER \
--subset-list $ORDER_CLASSIFIER_ACCEPTED_KEYS_PART2 \
--subset-key "orderKey"

# Print time taken to execute the script
echo "Time taken to verify images: $SECONDS seconds"
