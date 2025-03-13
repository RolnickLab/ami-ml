#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --output=train_classifier_%j.out

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
cp $ORDER_WBDS_ALL $SLURM_TMPDIR

echo "Time taken to copy the data: $((SECONDS/60)) minutes"

# 5. Launch your job
ami-classification train-model \
--model_type "convnext_tiny_in22k" \
--num_classes 16 \
--total_epochs 35 \
--early_stopping 35 \
--warmup_epochs 3 \
--train_webdataset "$SLURM_TMPDIR/train450-{000000..000364}.tar" \
--val_webdataset "$SLURM_TMPDIR/val450-{000000..000052}.tar" \
--test_webdataset "$SLURM_TMPDIR/test450-{000000..000105}.tar" \
--learning_rate_scheduler "cosine" \
--loss_function_type "weighted_order_and_binary_loss" \
--weight_on_order_loss 0 \
--model_save_directory $ORDER_CLASSIFIER_DATA_ON_SCRATCH \
--wandb_entity "moth-ai" \
--wandb_project "order-classifier" \
--wandb_run_name "worder0_wbinary1_run2"

# Print time taken to execute the script
echo "Time taken to train the model: $((SECONDS/60)) minutes"