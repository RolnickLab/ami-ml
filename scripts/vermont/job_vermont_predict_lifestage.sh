#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_lifestage
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --output=vermont_lifestage_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"
LIFESTAGE_MODEL="$BASE_DIR/models/lifestage_model"
LIFESTAGE_CATEGORY_MAP="$BASE_DIR/models/lifestage_category_map.json"

module load python/3.11

cd "$BASE_DIR"
source .venv/bin/activate

ami-dataset predict-lifestage \
  --verified-data-csv "$SPLITS/verification_results_clean.csv" \
  --results-csv "$SPLITS/lifestage_results.csv" \
  --wandb-run vermont_lifestage_prediction \
  --dataset-path "$IMAGES" \
  --model-path "$LIFESTAGE_MODEL" \
  --category-map-json "$LIFESTAGE_CATEGORY_MAP" \
  --wandb-entity "hack1996man" \
  --wandb-project "ai_for_leps" \
  --log-frequence 25 \
  --batch-size 1024 \
  --num-classes 2

echo "Life stage prediction completed at $(date)"
