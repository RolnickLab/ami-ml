#!/bin/bash
#SBATCH --job-name=tune_temperature_parameter
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx8000:1
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
cp $WEUROPE_CONF_CALIB_VAL_WBDS $WEUROPE_CONF_CALIB_TEST_WBDS $SLURM_TMPDIR

echo "Time taken to copy the data: $((SECONDS)) seconds"

# 5. Launch your job
python confidence_calibration/temperature_scaling.py \
--model-weights $WEUROPE_WEIGHTS \
--model-type resnet50 \
--num-classes 2603 \
--val-webdataset "$SLURM_TMPDIR/w-europe_val450-{000000..000238}.tar" \
--test-webdataset "$SLURM_TMPDIR/w-europe_test450-{000000..000476}.tar" \
--image-input-size 128 \
--batch-size 32 \
--preprocess-mode torch \
--trap-dataset-dir $CONF_CALIB_INSECT_CROPS_DIR \
--region WesternEurope \
--category-map $WEUROPE_CATEGORY_MAP

# Print time taken to execute the script
echo "Time taken to tune the temperature parameter: $((SECONDS/60)) minutes"