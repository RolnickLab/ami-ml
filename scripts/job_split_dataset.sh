#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=split_dataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=2:00:00
#SBATCH --output=split_dataset_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset split-dataset \
  --dataset-csv $FINAL_CLEAN_DATASET \
  --split-prefix $SPLIT_PREFIX \
  --max-instances 1000 \
  --min-instances 4

echo "Split completed at $(date)"
