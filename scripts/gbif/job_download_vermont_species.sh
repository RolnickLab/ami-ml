#!/bin/bash
#SBATCH --job-name=download_vermont_species
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=download_vermont_species_%j.out
#SBATCH --account=def-drolnick

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
OUT="$BASE_DIR/data/vermont_species.zip"

echo "Starting download at $(date)"
curl -L -C - --progress-bar \
  "https://api.gbif.org/v1/occurrence/download/request/0030333-260208012135463.zip" \
  -o "$OUT"

echo "Done at $(date)"
echo "File size: $(du -sh $OUT)"
