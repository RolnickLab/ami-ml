#!/bin/bash
#SBATCH --job-name=calculate_training_stats
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --partition=unkillable-cpu        # Ask for unkillable-cpu job
#SBATCH --cpus-per-task=1                 # Ask for 1 CPUs
#SBATCH --mem=4G                          # Ask for 4 GB of RAM
#SBATCH --output=calculate_training_stats_%j.out

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
python global_moth_model/calculate_training_stats.py

# Print time taken to execute the script
echo "Time taken to count the training images: $SECONDS seconds"
