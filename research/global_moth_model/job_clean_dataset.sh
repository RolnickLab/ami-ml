#!/bin/bash
#SBATCH --job-name=clean_dataset
#SBATCH --partition=long-cpu                # Ask for long-cpu job
#SBATCH --cpus-per-task=2                   # Ask for 2 CPUs
#SBATCH --mem=300G                          # Ask for 300 GB of RAM
#SBATCH --output=clean_dataset_%j.out

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
ami-dataset clean-dataset \
--dwca-file $DWCA_FILE \
--verified-data-csv $VERIFICATION_RESULTS \
--life-stage-predictions $LIFESTAGE_RESULTS 

# Print time taken to execute the script
echo "Time taken to clean the dataset: $SECONDS seconds"