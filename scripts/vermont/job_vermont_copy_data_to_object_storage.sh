#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_upload
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --output=vermont_upload_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DATA_DIR="$BASE_DIR/data/vermont_butterflies"
S3_DEST="s3://ami-datasets/vermont_butterflies"

cd "$BASE_DIR"
source .venv/bin/activate

aws s3 sync "$DATA_DIR" "$S3_DEST" --delete

echo "Upload completed at $(date)"
