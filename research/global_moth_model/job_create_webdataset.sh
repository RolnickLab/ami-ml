#!/bin/bash
#SBATCH --job-name=create_webdataset
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=long-cpu            # Ask for long-cpu job
#SBATCH --cpus-per-task=4               # Ask for 4 CPUs
#SBATCH --mem=10G                       # Ask for 10 GB of RAM
#SBATCH --output=create_webdataset_%j.out

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
ami-dataset create-webdataset \
--annotations-csv $SAMPLE_TRAIN_CSV \
--webdataset-pattern $SAMPLE_TRAIN_WBDS \
--wandb-run wbds_train_sample \
--dataset-path $GLOBAL_MODEL_DATASET_PATH \
--image-path-column image_path \
--label-column acceptedTaxonKey \
--columns-to-json $COLUMNS_TO_JSON \
--resize-min-size 450 \
--wandb-entity $WANDB_ENTITY \
--wandb-project $WANDB_PROJECT 
# --save-category-map-json $CATEGORY_MAP_JSON 


# Print time taken to execute the script
echo "Time taken to create the webdataset: $SECONDS seconds"
