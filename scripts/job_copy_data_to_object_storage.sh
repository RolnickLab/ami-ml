#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=upload_dataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --output=upload_dataset_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

aws s3 sync $GLOBAL_MODEL_DIR $GLOBAL_MODEL_OBJECT_STORE --delete

echo "Upload completed at $(date)"
