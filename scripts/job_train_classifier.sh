#!/bin/bash
#SBATCH --job-name=test_classifer_training_code
#SBATCH --ntasks=1
#SBATCH --time=14:00:00
#SBATCH --mem=48G
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=test_classifer_training_code_%j.out

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
cp $SAMPLE_TRAIN_WBDS_LINUX $SLURM_TMPDIR
cp $SAMPLE_VAL_WBDS_LINUX $SLURM_TMPDIR

echo "Time taken to copy the data: $((SECONDS/60)) minutes"

# 5. Launch your job #TODO:
ami-classification train-model \
--num_classes 29176 \
--train_webdataset "$SLURM_TMPDIR/train450-{000000..000976}.tar" \
--val_webdataset "$SLURM_TMPDIR/val450-{000000..000089}.tar" \
--test_webdataset "None" \
--model_save_directory $TEST_PATH

# Print time taken to execute the script
echo "Time taken to train the model: $((SECONDS/60)) minutes"