#!/bin/bash
#SBATCH --job-name=tune_temperature_parameter_global
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=140G
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=tune_temperature_parameter_global_%j.out

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
cp $GLOBAL_CONF_CALIB_TEST_WBDS $SLURM_TMPDIR

echo "Time taken to copy the data: $((SECONDS)) seconds"

# 5. Launch your job
python confidence_calibration/temperature_scaling.py \
--model-weights $GLOBAL_MODEL_WEIGHTS \
--model-type resnet50 \
--num-classes 29176 \
--val-webdataset "$SLURM_TMPDIR/test450-{000000..001357}.tar" \
--test-webdataset "$SLURM_TMPDIR/test450-{000000..001357}.tar" \
--image-input-size 128 \
--batch-size 32 \
--preprocess-mode torch \
--trap-dataset-dir $CONF_CALIB_INSECT_CROPS_DIR \
--region GLOBAL \
--category-map $CONF_CALIB_GLOBAL_MODEL_CATEGORY_MAP

# Print time taken to execute the script
echo "Time taken to tune the temperature parameter: $((SECONDS/60)) minutes"