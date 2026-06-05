#!/bin/bash
# Split a BQ-exported CSV into train/val/test CSVs.
# Reads from DATA_DIR/<CSV>, writes train.csv/val.csv/test.csv to DATA_DIR/splits/.
#
# Must run after job_bq_export.sh completes.
#
# Usage:
#   sbatch --dependency=afterok:<export_job_id> job_bq_split.sh
#   sbatch --export=CSV=global_min25occ.csv --dependency=afterok:<export_job_id> job_bq_split.sh
#
# CSV defaults to global_min25occ.csv if not set via --export.
#
#SBATCH --account=def-drolnick
#SBATCH --job-name=bq_split
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/bq_split_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

BASE_DIR="/project/6068129/melabbas/ami-ml"
DATA_DIR="${BASE_DIR}/data"
CSV="${CSV:-global_min25occ.csv}"
INPUT_CSV="${DATA_DIR}/${CSV}"
OUTPUT_DIR="${DATA_DIR}/splits"

echo "=== bq_split started at $(date) ==="
echo "Node       : $(hostname)"
echo "Input CSV  : ${INPUT_CSV}  ($(du -sh ${INPUT_CSV} | cut -f1))"
echo "Output dir : ${OUTPUT_DIR}"
echo ""

cd "${BASE_DIR}"
module load StdEnv/2023 arrow/17.0.0
source .venv/bin/activate

mkdir -p "${OUTPUT_DIR}"

python src/dataset_tools/bq_squashfs/split.py \
    --csv                 "${INPUT_CSV}" \
    --output-dir          "${OUTPUT_DIR}" \
    --category-key        species_name \
    --val-frac            0.1 \
    --test-frac           0.1 \
    --split-by-occurrence \
    --max-instances       1000 \
    --min-instances       5 \
    --seed                42

EXIT_CODE=$?
echo ""
echo "=== bq_split done at $(date) (exit=${EXIT_CODE}) ==="

if [ "${EXIT_CODE}" -eq 0 ]; then
    TRAIN_ROWS=$(( $(wc -l < "${OUTPUT_DIR}/train.csv") - 1 ))
    VAL_ROWS=$(( $(wc -l < "${OUTPUT_DIR}/val.csv") - 1 ))
    TEST_ROWS=$(( $(wc -l < "${OUTPUT_DIR}/test.csv") - 1 ))
    echo "  train : ${TRAIN_ROWS} rows"
    echo "  val   : ${VAL_ROWS} rows"
    echo "  test  : ${TEST_ROWS} rows"
    notify "bq_split: done" \
        "train=${TRAIN_ROWS}  val=${VAL_ROWS}  test=${TEST_ROWS}  → ${OUTPUT_DIR}"
else
    notify "bq_split: FAILED" \
        "exit=${EXIT_CODE} — check bq_split_${SLURM_JOB_ID}.out"
    exit 1
fi
