#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_verify
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=vermont_verify_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_butterflies.zip"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p "$SPLITS"

ami-dataset verify-images \
  --dwca-file "$DWCA" \
  --dataset-path "$IMAGES" \
  --results-csv "$SPLITS/verification_results.csv" \
  --num-workers 4

echo "Verify completed at $(date)"
