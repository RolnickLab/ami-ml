#!/bin/bash
#SBATCH --job-name=download_missing_images
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --partition=long-cpu         # Ask for long-cpu job
#SBATCH --cpus-per-task=1            # Ask for 1 CPUs
#SBATCH --mem=4G                     # Ask for 4 GB of RAM
#SBATCH --output=download_missing_images_%j.out

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
python confidence_calibration/download_missing_images.py