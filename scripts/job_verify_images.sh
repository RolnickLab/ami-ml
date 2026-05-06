#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=verify_gbif_images
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=verify_gbif_images_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset verify-images \
  --dataset-path $GLOBAL_MODEL_DATASET_PATH \
  --dwca-file $DWCA_FILE \
  --num-workers 16 \
  --results-csv $VERIFICATION_RESULTS \
  --resume-from-ckpt $VERIFICATION_RESULTS \
  --subset-list $ACCEPTED_KEY_LIST

echo "Verify completed at $(date)"
