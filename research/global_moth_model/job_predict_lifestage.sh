#!/bin/bash
#SBATCH --job-name=lifestage_prediction
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=long                       # Ask for long job
#SBATCH --cpus-per-task=8                      # Ask for 8 CPUs
#SBATCH --gres=gpu:rtx8000:1                   # Ask for 1 GPU
#SBATCH --output=lifestage_prediction_%j.out

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
ami-dataset predict-lifestage \
--dataset-path $GLOBAL_MODEL_DATASET_PATH \
--model-path $LIFESTAGE_MODEL \
--num-classes 2 \
--category-map-json $LIFESTAGE_CATEGORY_MAP \
--verified-data-csv $VERIFICATION_RESULTS \
--results-csv $LIFESTAGE_RESULTS \
--batch-size 256 

# Print time taken to execute the script
echo "Time taken to run life stage prediction: $SECONDS seconds"
