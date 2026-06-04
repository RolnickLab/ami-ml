#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=fetch_gbif_images
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=fetch_gbif_images_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset fetch-images \
  --dataset-path $GLOBAL_MODEL_DATASET_PATH \
  --dwca-file $DWCA_FILE \
  --num-images-per-category 1000 \
  --num-workers 4 \
  --subset-list $ACCEPTED_KEY_LIST

echo "Fetch completed at $(date)"
