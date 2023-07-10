#!/bin/bash
#SBATCH --partition=unkillable-cpu          # Ask for unkillable job
#SBATCH --cpus-per-task=2                   # Ask for 2 CPUs
#SBATCH --mem=4G                            # Ask for 4 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Download data
aws s3 sync s3://ami-trapdata/quebec/ ./quebec/
aws s3 sync s3://ami-trapdata/vermont/ ./vermont/