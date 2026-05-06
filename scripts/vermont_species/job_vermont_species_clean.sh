#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_species_clean
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=vermont_species_clean_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_gbif_download_v2/gbif_inat_vermont_butterflies_106spp_2479479rec_20260225.zip"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"

cd "$BASE_DIR"
source .venv/bin/activate

ami-dataset clean-dataset \
  --dwca-file "$DWCA" \
  --verified-data-csv "$SPLITS/verification_results.csv"

echo "Clean completed at $(date)"
