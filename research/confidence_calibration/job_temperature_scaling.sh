#!/bin/bash
#SBATCH --job-name=tune_temperature_parameter
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=tune_temperature_parameter_%j.out

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

# 4. Copy your dataset to the compute node
cp $CONF_CALIB_VAL_WBDS $SLURM_TMPDIR

echo "Time taken to copy the data: $((SECONDS)) seconds"

# 5. Launch your job
python confidence_calibration/temperature_scaling.py \
--model-weights $QUEBEC_VERMONT_WEIGHTS \
--model-type resnet50 \
--num-classes 2497 \
--val-webdataset "$SLURM_TMPDIR/ne-america_val450-{000000..000186}.tar" \
--image-input-size 128 \
--batch-size 8 \
--preprocess-mode torch

# Print time taken to execute the script
echo "Time taken to tune the temperature parameter: $((SECONDS/60)) minutes"