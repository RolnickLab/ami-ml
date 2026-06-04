#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_species_verify
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=vermont_species_verify_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_gbif_download_v2/gbif_inat_vermont_butterflies_106spp_2479479rec_20260225.zip"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p "$SPLITS"

RESUME_FLAG=""
if [[ -f "$SPLITS/verification_results.csv" ]]; then
  RESUME_FLAG="--resume-from-ckpt $SPLITS/verification_results.csv"
fi

ami-dataset verify-images \
  --dwca-file "$DWCA" \
  --dataset-path "$IMAGES" \
  --results-csv "$SPLITS/verification_results.csv" \
  $RESUME_FLAG \
  --num-workers 16

echo "Verify completed at $(date)"
