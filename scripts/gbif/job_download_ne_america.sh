#!/bin/bash
#SBATCH --job-name=download_ne_america
#SBATCH --account=def-drolnick
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/project/6068129/melabbas/data/ne-america-eccv2024/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

S5CMD="$HOME/bin/s5cmd"
ENDPOINT="https://object-arbutus.cloud.computecanada.ca"
DEST=/project/6068129/melabbas/data/ne-america-eccv2024/wbds/

mkdir -p "$DEST"

echo "Starting download at $(date)"
echo "Destination: $DEST"

# Build a run-file of only the missing shards (resumable)
RUNFILE=$(mktemp /tmp/s5cmd_runfile_XXXXXX.txt)

echo "Building run-file of missing files..."
S3_PREFIX="s3://ami-dataset-eccv2024/ami_gbif/fine-grained_classification/wbds"

$S5CMD --profile ami --endpoint-url "$ENDPOINT" \
    ls "${S3_PREFIX}/ne-america_*" \
    | awk '{print $NF}' \
    | while read fname; do
        if [ ! -f "$DEST/$fname" ]; then
            echo "cp ${S3_PREFIX}/${fname} $DEST/${fname}"
        fi
    done > "$RUNFILE"

N_MISSING=$(wc -l < "$RUNFILE")
echo "Files to download: $N_MISSING"

if [ "$N_MISSING" -gt 0 ]; then
    $S5CMD --profile ami --endpoint-url "$ENDPOINT" \
        --numworkers 32 \
        run "$RUNFILE"
else
    echo "All files already present, nothing to download."
fi

rm -f "$RUNFILE"

echo "Download complete at $(date)"

# Count and report
N_TRAIN=$(ls "$DEST"ne-america_train450-*.tar 2>/dev/null | wc -l)
N_VAL=$(ls   "$DEST"ne-america_val450-*.tar   2>/dev/null | wc -l)
N_TEST=$(ls  "$DEST"ne-america_test450-*.tar  2>/dev/null | wc -l)
TOTAL_GB=$(du -sh "$DEST" | cut -f1)

echo "train=$N_TRAIN val=$N_VAL test=$N_TEST total=$TOTAL_GB"

~/bin/notify "ne-america download done" \
    "train=${N_TRAIN}/1583, val=${N_VAL}/187, test=${N_TEST}/375, size=${TOTAL_GB} — $(date)"
