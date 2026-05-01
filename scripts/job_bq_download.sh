#!/bin/bash
# Download images from training_images BQ table in parallel.
# Runs as a SLURM array job — each task handles MOD(photo_id, NUM_JOBS) = task_id.
# After all tasks finish, run job_bq_pack_squashfs.sh to merge into a single SquashFS.
#
# Usage:
#   sbatch job_bq_download.sh
#   # or submit and chain the pack job:
#   DOWNLOAD_JOB=$(sbatch --parsable job_bq_download.sh)
#   sbatch --dependency=afterok:$DOWNLOAD_JOB job_bq_pack_squashfs.sh
#
#SBATCH --account=def-drolnick
#SBATCH --job-name=bq_download
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/bq_download_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

NUM_JOBS=10
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Each job downloads its images into its own staging subdirectory
# These are kept after the job ends (on Lustre) for the pack job to merge
STAGING_BASE="/scratch/melabbas/bq_download_staging"
STAGING_DIR="${STAGING_BASE}/task_${TASK_ID}"

echo "=== bq_download task=${TASK_ID}/${NUM_JOBS} started at $(date) ==="
echo "Node: $(hostname)"
echo "Staging dir: ${STAGING_DIR}"

mkdir -p "${STAGING_DIR}"

cd /project/6068129/melabbas/ami-ml
module load StdEnv/2023 arrow/17.0.0
source .venv/bin/activate

python src/dataset_tools/bq_squashfs/download_images.py \
    --staging-dir  "${STAGING_DIR}" \
    --num-jobs     ${NUM_JOBS} \
    --task-id      ${TASK_ID} \
    --num-workers  32 \
    --chunk-size   10000

EXIT_CODE=$?
echo "=== bq_download task=${TASK_ID} finished at $(date) (exit=${EXIT_CODE}) ==="

notify "bq_download task ${TASK_ID}: done" \
    "Staging: ${STAGING_DIR} | exit=${EXIT_CODE}"
