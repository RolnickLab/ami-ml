#!/bin/bash
#SBATCH --job-name=fetch_data_from_object_store
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --partition=unkillable-cpu      # Ask for unkillable-cpu job
#SBATCH --cpus-per-task=2               # Ask for 2 CPUs
#SBATCH --mem=4G                        # Ask for 4 GB of RAM
#SBATCH --output=fetch_data_from_object_store_%j.out

## Run this from the research directory.
## ./research/<sub_dir(s)>/<filename>.sh

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
aws s3 sync $FINAL_AMITRAPS_DATA_OBJECT_STORE $PERSONAL_SCRATCH

# Print time taken to execute the script
echo "Time taken to download the data: $SECONDS seconds"
