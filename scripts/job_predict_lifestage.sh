#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=lifestage_prediction
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --output=lifestage_prediction_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

module load python/3.11

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset predict-lifestage \
  --verified-data-csv $VERIFICATION_RESULTS_P2 \
  --results-csv $LIFESTAGE_RESULTS_P2 \
  --wandb-run lifestage_prediction_p2 \
  --dataset-path $GLOBAL_MODEL_DATASET_PATH \
  --model-path $LIFESTAGE_MODEL \
  --category-map-json $LIFESTAGE_CATEGORY_MAP \
  --wandb-entity $WANDB_ENTITY \
  --wandb-project $WANDB_PROJECT \
  --log-frequence 25 \
  --batch-size 1024 \
  --num-classes 2

echo "Life stage prediction completed at $(date)"
