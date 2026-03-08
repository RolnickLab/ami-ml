#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_delete_images
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=vermont_delete_images_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"

cd "$BASE_DIR"
source .venv/bin/activate

ami-dataset delete-images \
  --error-images-csv "$SPLITS/verification_errors.csv" \
  --base-path "$IMAGES"

echo "Delete completed at $(date)"
