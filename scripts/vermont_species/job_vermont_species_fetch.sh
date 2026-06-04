#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_species_fetch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=vermont_species_fetch_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_gbif_download_v2/gbif_inat_vermont_butterflies_106spp_2479479rec_20260225.zip"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"
MAX_IMAGES_PER_SPECIES=1000   # set to 0 for no limit

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p "$IMAGES"

# Already-downloaded images are skipped automatically (tool checks if file exists)
ami-dataset fetch-images \
  --dwca-file "$DWCA" \
  --dataset-path "$IMAGES" \
  --num-workers 8 \
  --num-images-per-category "$MAX_IMAGES_PER_SPECIES" \
  --subset-key verbatimScientificName

echo "Fetch completed at $(date)"
