#!/bin/bash
#SBATCH --job-name=upload_dataset
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --partition=long-cpu          # Ask for long cpu job
#SBATCH --cpus-per-task=2             # Ask for 2 CPUs
#SBATCH --mem=4G                      # Ask for 4 GB of RAM
#SBATCH --output=upload_dataset_%j.out

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
aws s3 sync $ORDER_CLASSIFIER_DATA_ON_SCRATCH $ORDER_CLASSIFIER_DATA_ON_OBJECT_STORE --delete

# Print time taken to execute the script
echo "Time taken to upload the dataset: $SECONDS seconds"
