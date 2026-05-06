#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=inspect_webdataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/test/inspect_webdataset_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
WBDS_DIR="/project/6068129/melabbas/ami-ml/data/vermont_butterflies/webdataset"
# Category map auto-discovered from WBDS_DIR/*_category_map.json — no need to specify manually
# unless auto-discovery fails (0 or multiple *_category_map.json files found)

cd "$BASE_DIR"
source .venv/bin/activate

python src/dataset_tools/inspect_webdataset.py \
  --webdataset-dir "$WBDS_DIR" \
  --split vermont_butterflies_val_verbatim \
  --sort-by count

~/bin/notify "inspect_webdataset done" "See job output for class distribution: inspect_webdataset_${SLURM_JOB_ID}.out"
