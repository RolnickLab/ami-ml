#!/bin/bash
#SBATCH --job-name=fine_tune_classifier
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --output=fine_tune_classifier_%j.out

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

# 4. Copy the dataset to the compute node
cp $FINE_TUNING_UK_DENMARK_WBDS_ALL $SLURM_TMPDIR
ls $SLURM_TMPDIR
echo "Time taken to copy the data: $((SECONDS/60)) minutes"

# 5. Launch your job
ami-classification train-model \
--model_type "convnext_tiny_in22k" \
--num_classes 2603 \
--existing_weights $WEUROPE_CONVNEXT_T_WEIGHTS \
--total_epochs 15 \
--early_stopping 15 \
--warmup_epochs 3 \
--train_webdataset "$SLURM_TMPDIR/train-000000.tar" \
--val_webdataset "$SLURM_TMPDIR/val-000000.tar" \
--test_webdataset "$SLURM_TMPDIR/test-000000.tar" \
--batch_size 16 \
--learning_rate_scheduler "cosine" \
--mixed_resolution_data_aug False \
--model_save_directory $FINE_TUNING_UK_DENMARK_DATA_DIR \
--wandb_entity "moth-ai" \
--wandb_project "fine-tuning" \
--wandb_run_name "fine-tuning_uk-denmark_v0"

# Print time taken to execute the script
echo "Time taken to train the model: $((SECONDS/60)) minutes"