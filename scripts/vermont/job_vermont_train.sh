#!/bin/bash
#SBATCH --job-name=vermont_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=vermont_train_%j.out
#SBATCH --partition=gpubase_bynode_b1

module load python/3.11

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
WBDS="$BASE_DIR/data/vermont_butterflies/webdataset"
MODELS="$BASE_DIR/data/vermont_butterflies/models"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p "$MODELS"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-classification train-model \
  --train_webdataset "$WBDS/vermont_train_verbatim-{000000..000051}.tar" \
  --val_webdataset "$WBDS/vermont_val_verbatim-{000000..000014}.tar" \
  --test_webdataset "$WBDS/vermont_test_verbatim-{000000..000014}.tar" \
  --num_classes 133 \
  --model_save_directory "$MODELS" \
  --total_epochs 20 \
  --batch_size 16 \
  --wandb_entity "hack1996man" \
  --wandb_project "ai_for_leps" \
  --wandb_run_name "vermont_verbatim_run1"

echo "Training completed at $(date)"
