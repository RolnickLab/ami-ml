#!/bin/bash
#SBATCH --job-name=lifestage_prediction
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=long     # Ask for long job
#SBATCH --cpus-per-task=4    # Ask for 4 CPUs
#SBATCH --gres=gpu:1         # Ask or 1 GPU
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
--verified-data-csv $VERIFICATION_RESULTS_P2 \
--results-csv $LIFESTAGE_RESULTS_P2 \
--wandb-run lifestage_prediction_p2 \
--dataset-path $GLOBAL_MODEL_DATASET_PATH \
--model-path $LIFESTAGE_MODEL \
--category-map-json $LIFESTAGE_CATEGORY_MAP \
--wandb-entity $WANDB_ENTITY \
--wandb-project $WANDB_PROJECT \
--log-frequence 25 \
--batch-size 1024 \
--num-classes 2

# Print time taken to execute the script
echo "Time taken to run life stage prediction: $SECONDS seconds"
