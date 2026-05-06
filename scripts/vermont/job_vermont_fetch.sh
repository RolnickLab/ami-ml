#!/bin/bash
#SBATCH --job-name=vermont_fetch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=vermont_fetch_%j.out

# Paths - adjust if needed
BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_butterflies.zip"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p "$IMAGES"

ami-dataset fetch-images \
  --dwca-file "$DWCA" \
  --dataset-path "$IMAGES" \
  --num-workers 4

echo "Fetch completed at $(date)"
