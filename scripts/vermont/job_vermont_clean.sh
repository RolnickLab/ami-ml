#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_clean
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=vermont_clean_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_butterflies.zip"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"

cd "$BASE_DIR"
source .venv/bin/activate

ami-dataset clean-dataset \
  --dwca-file "$DWCA" \
  --verified-data-csv "$SPLITS/verification_results.csv"

echo "Clean completed at $(date)"
