#!/bin/bash
#SBATCH --job-name=fetch_gbif_images
#SBATCH --partition=long-cpu              # Ask for long-cpu job
#SBATCH --cpus-per-task=16                # Ask for 16 CPUs
#SBATCH --mem=300G                        # Ask for 300 GB of RAM
#SBATCH --output=fetch_gbif_images_%j.out

# NOTE: The current workflow loads the entire DwC-A file as is, hence the big memory requirement (300G).
# This can definitely be improved.

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
--num-images-per-category 6000 \
--num-workers 16 \
--subset-list $ORDER_CLASSIFIER_KEYS_HYMENOPTERA_FAMILIES \
--subset-key "familyKey"


# Print time taken to execute the script
echo "Time taken to fetch images: $SECONDS seconds"
