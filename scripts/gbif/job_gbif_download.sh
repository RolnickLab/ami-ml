#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=gbif_download
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=gbif_download_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
NAMES_FILE="$BASE_DIR/data/vermont_species_list_v2.txt"
OUTPUT_DIR="$BASE_DIR/data/vermont_gbif_download_v2"
COUNTRY=""        # e.g. "US" — leave empty to omit
BBOX=""           # e.g. "42.7,-73.5,45.0,-71.5" — leave empty to omit
DATASET_KEY="50c9509d-22c7-4a22-a47d-8c48425ef4a7"

cd "$BASE_DIR"
source .venv/bin/activate
if [[ -f .env ]]; then set -a; source .env; set +a; fi  # loads GBIF_USER/PWD/EMAIL
mkdir -p "$OUTPUT_DIR"

EXTRA_FLAGS=""
[[ -n "$COUNTRY" ]]     && EXTRA_FLAGS="$EXTRA_FLAGS --country $COUNTRY"
[[ -n "$BBOX" ]]        && EXTRA_FLAGS="$EXTRA_FLAGS --bbox $BBOX"
[[ -n "$DATASET_KEY" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --dataset-key $DATASET_KEY"

ami-dataset download-gbif \
    --names-file "$NAMES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --poll-interval 30 \
    --max-wait 14400 \
    $EXTRA_FLAGS

echo "Done at $(date)"; ls -lh "$OUTPUT_DIR"
