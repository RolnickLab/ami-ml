#!/bin/bash
# Export a BigQuery query to CSV.
#
# Usage:
#   sbatch --export=QUERY_FILE=queries/global_min25occ.sql,OUTPUT=global_min25occ.csv  job_bq_export.sh
#   sbatch --export=QUERY_FILE=queries/global_max2000img.sql,OUTPUT=global_max2000img.csv job_bq_export.sh
#
#SBATCH --account=def-drolnick
#SBATCH --job-name=bq_export
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/bq_export_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DATA_DIR="/project/6068129/melabbas/ami-ml/data"

echo "=== bq_export started at $(date) ==="
echo "Node       : $(hostname)"
echo "QUERY_FILE : ${QUERY_FILE}"
echo "OUTPUT     : ${DATA_DIR}/${OUTPUT}"
echo ""

cd "${BASE_DIR}"
module load StdEnv/2023 arrow/17.0.0
source .venv/bin/activate

python src/dataset_tools/bq_squashfs/bq_export.py \
    --query-file "src/dataset_tools/bq_squashfs/${QUERY_FILE}" \
    --output     "${DATA_DIR}/${OUTPUT}" \
    --project    leps-ai

EXIT_CODE=$?
echo ""
echo "=== bq_export done at $(date) (exit=${EXIT_CODE}) ==="

if [ "${EXIT_CODE}" -eq 0 ]; then
    SIZE=$(du -sh "${DATA_DIR}/${OUTPUT}" | cut -f1)
    ROWS=$(( $(wc -l < "${DATA_DIR}/${OUTPUT}") - 1 ))
    notify "bq_export done" "${OUTPUT}  ${ROWS} rows  ${SIZE}"
else
    notify "bq_export FAILED" "exit=${EXIT_CODE} — check bq_export_${SLURM_JOB_ID}.out"
    exit 1
fi
