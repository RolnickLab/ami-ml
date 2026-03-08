#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=clean_dataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=clean_dataset_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

ami-dataset clean-dataset \
  --dwca-file $DWCA_FILE \
  --verified-data-csv $VERIFICATION_RESULTS \
  --life-stage-predictions $LIFESTAGE_RESULTS

echo "Clean completed at $(date)"
