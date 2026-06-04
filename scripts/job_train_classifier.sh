#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=train_classifier
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --output=train_classifier_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

module load python/3.11

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Copy dataset to compute node for faster I/O
cp $SAMPLE_WBDS_LINUX $SLURM_TMPDIR

ami-classification train-model \
  --num_classes 2497 \
  --train_webdataset "$SLURM_TMPDIR/ne-america_train450-{000000..000010}.tar" \
  --val_webdataset "$SLURM_TMPDIR/ne-america_val450-{000000..000015}.tar" \
  --test_webdataset "$SLURM_TMPDIR/ne-america_test450-{000000..000005}.tar" \
  --model_save_directory $TEST_PATH \
  --total_epochs 20 \
  --wandb_entity moth-ai \
  --wandb_project test

echo "Training completed at $(date)"
