#!/bin/bash
#SBATCH --job-name=delete_corrupted_images
#SBATCH --partition=long-cpu              # Ask for long-cpu job
#SBATCH --cpus-per-task=2                 # Ask for 2 CPUs
#SBATCH --mem=2G                          # Ask for 2 GB of RAM
#SBATCH --output=delete_corrupted_images_%j.out

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
ami-dataset delete-images \
--error-images-csv $VERIFICATION_ERROR_RESULTS \
--base-path $GLOBAL_MODEL_DATASET_PATH

# Print time taken to execute the script
echo "Time taken to delete the corrupted images: $SECONDS seconds"
