#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=create_webdataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --output=create_webdataset_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset create-webdataset \
  --annotations-csv $TEST_CSV \
  --webdataset-pattern $TEST_WBDS \
  --wandb-run wbds_test \
  --dataset-path $GLOBAL_MODEL_DATASET_PATH \
  --image-path-column image_path \
  --label-column acceptedTaxonKey \
  --columns-to-json $COLUMNS_TO_JSON \
  --resize-min-size 450 \
  --category-map-json $CATEGORY_MAP_JSON \
  --wandb-entity $WANDB_ENTITY \
  --wandb-project $WANDB_PROJECT

echo "Webdataset completed at $(date)"
