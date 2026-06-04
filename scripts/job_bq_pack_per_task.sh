#!/bin/bash
# Merge per-chunk SquashFS files for ONE task into a single task-level SquashFS.
#
# Runs as a SLURM array job — each array task handles one download task_N.
# After all tasks complete, task_0.sqfs … task_9.sqfs are ready for use
# by the webdataset build job.
#
# Safety design:
#   1. Stream all chunks → sqfstar → task_N.sqfs   (no deletion during stream)
#   2. Verify: count images in output sqfs == sum of images across all chunks
#   3. Only delete chunks if and only if verification passes
#
# This guarantees zero data loss: if sqfstar OOMs or the job times out, all
# chunk files are preserved and the job can be resubmitted with more --mem
# or --time without redownloading anything.
#
# Usage:
#   sbatch --array=0-9 job_bq_pack_per_task.sh
#   # or chain after download:
#   DOWNLOAD_JOB=$(sbatch --parsable --array=0-9 job_bq_download.sh)
#   sbatch --array=0-9 --dependency=afterok:$DOWNLOAD_JOB job_bq_pack_per_task.sh
#
#SBATCH --account=def-drolnick
#SBATCH --job-name=bq_pack_task
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/bq_pack_task_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_DIR="/scratch/melabbas/bq_download_staging/task_${TASK_ID}"
OUTPUT_SQFS="/scratch/melabbas/task_${TASK_ID}.sqfs"

echo "=== bq_pack_task ${TASK_ID} started at $(date) ==="
echo "Node     : $(hostname)"
echo "Task dir : ${TASK_DIR}"
echo "Output   : ${OUTPUT_SQFS}"
echo ""

# ── Pre-flight ────────────────────────────────────────────────────────────────

TOTAL_CHUNKS=$(find "${TASK_DIR}" -name "chunk_*.sqfs" 2>/dev/null | wc -l)
if [ "${TOTAL_CHUNKS}" -eq 0 ]; then
    echo "No chunks found for task ${TASK_ID} — nothing to merge."
    notify "bq_pack_task ${TASK_ID}: skipped" "No chunks found in ${TASK_DIR}"
    exit 0
fi
echo "Chunks to merge: ${TOTAL_CHUNKS}"
echo ""

# Count expected images across all chunks (metadata read only, no extraction)
echo "Counting expected images across all chunks..."
EXPECTED_IMAGES=0
for chunk in $(find "${TASK_DIR}" -name "chunk_*.sqfs" | sort); do
    COUNT=$(unsquashfs -l "${chunk}" 2>/dev/null | grep -cE '\.(jpg|jpeg|png)$' || echo 0)
    echo "  $(basename ${chunk}): ${COUNT} images"
    EXPECTED_IMAGES=$((EXPECTED_IMAGES + COUNT))
done
echo "Expected total: ${EXPECTED_IMAGES} images"
echo ""

# ── Merge ─────────────────────────────────────────────────────────────────────

rm -f "${OUTPUT_SQFS}"

cd /project/6068129/melabbas/ami-ml
module load StdEnv/2023 arrow/17.0.0
source .venv/bin/activate

echo "=== Streaming chunks → sqfstar at $(date) ==="

python src/dataset_tools/bq_squashfs/merge_sqfs_chunks.py \
    "${TASK_DIR}" \
  | sqfstar \
        -comp zstd \
        -Xcompression-level 3 \
        -b 131072 \
        -no-duplicates \
        "${OUTPUT_SQFS}"

# Capture atomically — assigning PIPESTATUS[0] to a variable resets PIPESTATUS,
# so both values must be saved in a single array assignment first.
PIPE_STATUS=("${PIPESTATUS[@]}")
STREAM_EXIT="${PIPE_STATUS[0]}"
SQFSTAR_EXIT="${PIPE_STATUS[1]}"

echo ""
echo "=== Merge finished at $(date) ==="
echo "stream_chunks_to_tar exit : ${STREAM_EXIT}"
echo "sqfstar exit              : ${SQFSTAR_EXIT}"
echo ""

# ── Verify ────────────────────────────────────────────────────────────────────

if [ "${STREAM_EXIT}" -ne 0 ] || [ "${SQFSTAR_EXIT}" -ne 0 ]; then
    echo "ERROR: merge failed (stream=${STREAM_EXIT} sqfstar=${SQFSTAR_EXIT})"
    echo "Chunks preserved in ${TASK_DIR} — re-submit with more --mem or investigate errors."
    notify "bq_pack_task ${TASK_ID}: FAILED" \
        "stream=${STREAM_EXIT} sqfstar=${SQFSTAR_EXIT} — chunks preserved, re-submit"
    exit 1
fi

if [ ! -f "${OUTPUT_SQFS}" ]; then
    echo "ERROR: output sqfs not found at ${OUTPUT_SQFS}"
    notify "bq_pack_task ${TASK_ID}: FAILED" "output sqfs missing — chunks preserved"
    exit 1
fi

echo "Verifying output sqfs image count..."
ACTUAL_IMAGES=$(unsquashfs -l "${OUTPUT_SQFS}" 2>/dev/null | grep -cE '\.(jpg|jpeg|png)$' || echo 0)
SIZE=$(du -sh "${OUTPUT_SQFS}" | cut -f1)

echo "  Expected : ${EXPECTED_IMAGES} images"
echo "  Actual   : ${ACTUAL_IMAGES} images"
echo "  Size     : ${SIZE}"
echo ""

if [ "${ACTUAL_IMAGES}" -ne "${EXPECTED_IMAGES}" ]; then
    echo "ERROR: image count mismatch (expected=${EXPECTED_IMAGES} actual=${ACTUAL_IMAGES})"
    echo "Output sqfs may be incomplete. Chunks preserved in ${TASK_DIR}."
    echo "Investigate: run audit_sqfs.py or check stream_chunks_to_tar logs above."
    notify "bq_pack_task ${TASK_ID}: FAILED (count mismatch)" \
        "expected=${EXPECTED_IMAGES} actual=${ACTUAL_IMAGES} — chunks preserved in ${TASK_DIR}"
    exit 1
fi

echo "Verification passed: ${ACTUAL_IMAGES} images confirmed in ${OUTPUT_SQFS}"
echo ""

# ── Safe delete ───────────────────────────────────────────────────────────────
# Only reached when: stream_exit=0, sqfstar_exit=0, image count matches.

echo "Deleting chunk files (verified, safe to remove)..."
DELETED=0
for chunk in $(find "${TASK_DIR}" -name "chunk_*.sqfs" | sort); do
    rm -f "${chunk}"
    DELETED=$((DELETED + 1))
done
echo "Deleted ${DELETED} chunk files from ${TASK_DIR}"
echo ""

echo "=== bq_pack_task ${TASK_ID} done at $(date) ==="
notify "bq_pack_task ${TASK_ID}: done" \
    "${ACTUAL_IMAGES} images in ${OUTPUT_SQFS} (${SIZE}) — ${DELETED} chunks deleted"
