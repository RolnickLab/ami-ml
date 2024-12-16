#!/bin/bash
#SBATCH --job-name=create_webdataset_test
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=long-cpu            # Ask for long-cpu job
#SBATCH --cpus-per-task=4               # Ask for 4 CPUs
#SBATCH --mem=10G                       # Ask for 10 GB of RAM
#SBATCH --output=create_webdataset_test_%j.out

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
--annotations-csv $ORDER_TEST_CSV \
--webdataset-pattern $ORDER_TEST_WBDS \
--dataset-path $ORDER_CLASSIFIER_RAW_DATA \
--image-path-column "image_path" \
--label-column "orderKey" \
--columns-to-json $COLUMNS_TO_JSON \
--resize-min-size 450 \
--category-map-json $ORDER_CATEGORY_MAP 


# Print time taken to execute the script
echo "Time taken to create the webdataset: $SECONDS seconds"
