#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_species_split
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0:30:00
#SBATCH --output=vermont_species_split_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"

cd "$BASE_DIR"
source .venv/bin/activate

ami-dataset split-dataset \
  --dataset-csv "$SPLITS/verification_results_clean.csv" \
  --split-prefix "$SPLITS" \
  --max-instances 1000 \
  --min-instances 0 \
  --test-frac 0.2 \
  --val-frac 0.2 \
  --category-key verbatimScientificName

echo "Split completed at $(date)"
