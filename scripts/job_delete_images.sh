#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=delete_corrupted_images
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=delete_corrupted_images_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset delete-images \
  --error-images-csv $VERIFICATION_ERROR_RESULTS \
  --base-path $GLOBAL_MODEL_DATASET_PATH

echo "Delete completed at $(date)"
