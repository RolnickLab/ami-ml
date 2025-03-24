#!/bin/bash
#SBATCH --job-name=fetch_data_from_object_store
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --partition=main-cpu      # Ask for long-cpu job
#SBATCH --cpus-per-task=1         # Ask for 1 CPUs
#SBATCH --mem=4G                  # Ask for 4 GB of RAM
#SBATCH --output=fetch_data_from_object_store_%j.out

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

# 4. Download data
aws s3 sync $OBJECT_STORE_GLOBAL_MODEL_RAW_DATA $CONF_CALIB_GLOBAL_MODEL_DATASET_PATH 

# Print time taken to execute the script
echo "Time taken to download the data: $SECONDS seconds"