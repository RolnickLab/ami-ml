#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=gbif_download_northamerica
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=gbif_download_northamerica_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
NAMES_FILE="$BASE_DIR/data/vermont_species_list_v2.txt"
OUTPUT_DIR="$BASE_DIR/data/northamerica_gbif_download_v1"
DATASET_KEY="50c9509d-22c7-4a22-a47d-8c48425ef4a7"

cd "$BASE_DIR"
source .venv/bin/activate
if [[ -f .env ]]; then set -a; source .env; set +a; fi  # loads GBIF_USER/PWD/EMAIL
mkdir -p "$OUTPUT_DIR"

ami-dataset download-gbif \
    --names-file "$NAMES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-key "$DATASET_KEY" \
    --country US \
    --country CA \
    --poll-interval 60 \
    --max-wait 21600

echo "Done at $(date)"; ls -lh "$OUTPUT_DIR"
notify "gbif northamerica download done" "DwCA for 106 Vermont spp (US+CA, iNat) saved to $OUTPUT_DIR"
