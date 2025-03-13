#!/bin/bash
#SBATCH --job-name=split_dataset
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --partition=long-cpu              # Ask for long-cpu job
#SBATCH --cpus-per-task=2                 # Ask for 2 CPUs
#SBATCH --mem=6G                          # Ask for 6 GB of RAM
#SBATCH --output=split_dataset_%j.out

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
ami-dataset split-dataset \
--dataset-csv $ORDER_CLEAN_DATASET \
--split-prefix $ORDER_CLASSIFIER_DATA_ON_SCRATCH \
--category-key "orderKey" \
--max-instances -1 \
--min-instances 0

# Print time taken to execute the script
echo "Time taken to split the dataset: $SECONDS seconds"
