#!/bin/bash
#SBATCH --account=def-drolnick_gpu
#SBATCH --job-name=vermont_species_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=vermont_species_train_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com
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

# Dynamically resolve shard counts and num_classes
N_TRAIN=$(ls "$WBDS"/vermont_butterflies_train_verbatim-*.tar 2>/dev/null | wc -l)
N_VAL=$(ls "$WBDS"/vermont_butterflies_val_verbatim-*.tar 2>/dev/null | wc -l)
N_TEST=$(ls "$WBDS"/vermont_butterflies_test_verbatim-*.tar 2>/dev/null | wc -l)
NUM_CLASSES=$(python -c "import json; d=json.load(open('$WBDS/vermont_butterflies_category_map.json')); print(len(d))")

TRAIN_PAT="$WBDS/vermont_butterflies_train_verbatim-{000000..$(printf '%06d' $((N_TRAIN-1)))}.tar"
VAL_PAT="$WBDS/vermont_butterflies_val_verbatim-{000000..$(printf '%06d' $((N_VAL-1)))}.tar"
TEST_PAT="$WBDS/vermont_butterflies_test_verbatim-{000000..$(printf '%06d' $((N_TEST-1)))}.tar"

echo "Train shards: $N_TRAIN, Val shards: $N_VAL, Test shards: $N_TEST, Num classes: $NUM_CLASSES"

ami-classification train-model \
  --train_webdataset "$TRAIN_PAT" \
  --val_webdataset "$VAL_PAT" \
  --test_webdataset "$TEST_PAT" \
  --num_classes "$NUM_CLASSES" \
  --model_save_directory "$MODELS" \
  --total_epochs 30 \
  --warmup_epochs 2 \
  --batch_size 128 \
  --random_seed 123 \
  --wandb_entity "hack1996man" \
  --wandb_project "ai_for_leps" \
  --wandb_run_name "vermont_butterflies_verbatim_run2"

echo "Training completed at $(date)"
