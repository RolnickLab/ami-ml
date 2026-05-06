#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=download_quebec_vermont
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/download_quebec_vermont_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

DEST="/project/6068129/melabbas/data/quebec_vermont"
ENDPOINT="https://object-arbutus.cloud.computecanada.ca"
SRC="s3://ami-fine-tuning/quebec_vermont"

export AWS_PROFILE=ami
export AWS_REGION=us-east-1

mkdir -p "$DEST"

echo "Starting download at $(date)"

s5cmd --endpoint-url "$ENDPOINT" cp --if-size-differ \
    "${SRC}/*" "$DEST/"

echo "Download completed at $(date)"

~/bin/notify "download_quebec_vermont: done" "Dataset downloaded to $DEST"
