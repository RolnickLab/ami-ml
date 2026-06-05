#!/bin/bash
# =============================================================================
# job_bq_download.sh — staged BQ image download → SquashFS → object store
# =============================================================================
#
# WHAT IT DOES
# ------------
# Downloads every pending image from a BigQuery training_images table and
# delivers it to the Arbutus object store as one SquashFS archive per array
# task, using only a small, bounded amount of scratch space.
#
# Runs as a SLURM array job of NUM_JOBS tasks. Each task owns the dataset
# slice  MOD(photo_id, NUM_JOBS) == SLURM_ARRAY_TASK_ID  and performs the
# FULL lifecycle for that slice before releasing its scratch space:
#
#   stage 1  download pending images (32 threads, BQ-checkpointed every
#            10k-image chunk, each chunk packed to chunk_NNNN.sqfs)
#   stage 2  count expected images across all chunk files (unsquashfs -l)
#   stage 3  stream-merge chunks into a single task_N.sqfs (sqfstar, zstd)
#   stage 4  verify merged image count == stage 2 count
#   stage 5  upload task_N.sqfs to the object store, verify byte size
#   stage 6  delete local chunks + merged sqfs (only after stage 5 verifies)
#
# Because tasks clean up after themselves, peak scratch usage is
#   (concurrent tasks) x (~2x final task sqfs size)
# and is controlled by the array throttle (%K below), NOT the dataset size.
#
# STATUS TRACKING (BigQuery)
# --------------------------
# Per chunk, results are appended to <dataset>.training_images_downloads and
# MERGEd into <dataset>.training_images (fetch_status = downloaded / failed /
# corrupted + image dims). The table is therefore updated live as the job
# runs. Concurrent MERGE serialization conflicts are retried with backoff
# inside download_images.py.
#
# FAILURE / RESUME SEMANTICS
# --------------------------
# Nothing local is deleted until the uploaded archive is verified, and any
# failure notifies (ntfy) and exits non-zero. Resubmitting a failed or
# timed-out task id is always safe:
#   - if the remote task_N.sqfs already exists, the task exits 0 immediately
#   - already-downloaded images are skipped via training_images_downloads
#   - chunk files from a previous partial run are stashed into a resume_*
#     subdir (the downloader restarts chunk numbering and would overwrite
#     them; the merge step searches recursively and still includes them)
#
# USAGE
# -----
#   sbatch scripts/job_bq_download.sh                 # full run
#   sbatch --array=7 scripts/job_bq_download.sh      # re-run one task
#   scontrol update JobId=<id> ArrayTaskThrottle=8    # change concurrency live
#
# Before pointing at a new dataset, update STAGING_BASE, S3_DEST and
# --dataset below, and check scratch headroom vs the %K throttle.
# =============================================================================
#
#SBATCH --account=def-drolnick
#SBATCH --job-name=bq_dl_2605
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-59%4
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/bq_download_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -uo pipefail

NUM_JOBS=60
TASK_ID=${SLURM_ARRAY_TASK_ID}

STAGING_BASE="/scratch/melabbas/global_all_leps_2605"
TASK_DIR="${STAGING_BASE}/task_${TASK_ID}"
OUT_SQFS="${STAGING_BASE}/task_${TASK_ID}.sqfs"
S3_DEST="s3://ami-trainingdata/ai-for-leps/global_all_leps_2605/task_${TASK_ID}.sqfs"
ENDPOINT="https://object-arbutus.cloud.computecanada.ca"
export AWS_PROFILE=ami

fail() {
    echo "ERROR: $1"
    notify "bq_download task ${TASK_ID}: FAILED" "$1 | logs: bq_download_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
    exit 1
}

echo "=== bq_download task=${TASK_ID}/${NUM_JOBS} started $(date) on $(hostname) ==="

# ── Idempotence: skip if this task already completed ─────────────────────────
if s5cmd --endpoint-url "${ENDPOINT}" ls "${S3_DEST}" >/dev/null 2>&1; then
    echo "Remote ${S3_DEST} already exists — task previously completed. Exiting."
    exit 0
fi

mkdir -p "${TASK_DIR}"

# ── Resume safety: stash chunks from a previous partial run ──────────────────
# download_images.py restarts numbering at chunk_0001 and would overwrite them;
# merge_sqfs_chunks.py searches recursively, so a subdir keeps them mergeable.
if compgen -G "${TASK_DIR}/chunk_*.sqfs" > /dev/null; then
    RESUME_DIR="${TASK_DIR}/resume_${SLURM_ARRAY_JOB_ID}"
    mkdir -p "${RESUME_DIR}"
    mv "${TASK_DIR}"/chunk_*.sqfs "${RESUME_DIR}/"
    echo "Resume: moved $(ls ${RESUME_DIR} | wc -l) existing chunks to ${RESUME_DIR}"
fi

cd /project/6068129/melabbas/ami-ml
module load StdEnv/2023 arrow/17.0.0
source .venv/bin/activate

# ── STAGE 1: download + pack chunks ──────────────────────────────────────────
echo "=== STAGE 1: download (num_jobs=${NUM_JOBS}, chunk_size=10000) $(date) ==="
T0=$SECONDS
python src/dataset_tools/bq_squashfs/download_images.py \
    --staging-dir  "${TASK_DIR}" \
    --num-jobs     ${NUM_JOBS} \
    --task-id      ${TASK_ID} \
    --num-workers  32 \
    --chunk-size   10000 \
    --dataset      global_all_leps_2605 \
    || fail "download_images.py exited non-zero"
echo "STAGE 1 done in $((SECONDS - T0))s"

# ── STAGE 2: count expected images across all chunks (incl. resume_* dirs) ──
echo "=== STAGE 2: count expected images $(date) ==="
EXPECTED=0
while IFS= read -r chunk; do
    N=$(unsquashfs -l "${chunk}" 2>/dev/null | grep -cE '\.(jpg|jpeg|png)$' || echo 0)
    EXPECTED=$((EXPECTED + N))
done < <(find "${TASK_DIR}" -name "chunk_*.sqfs")
echo "Expected: ${EXPECTED} images in $(find "${TASK_DIR}" -name 'chunk_*.sqfs' | wc -l) chunks"
[ "${EXPECTED}" -gt 0 ] || fail "no images found in chunks"

# ── STAGE 3: merge chunks → task sqfs ────────────────────────────────────────
echo "=== STAGE 3: merge $(date) ==="
T0=$SECONDS
rm -f "${OUT_SQFS}"
python src/dataset_tools/bq_squashfs/merge_sqfs_chunks.py "${TASK_DIR}" \
  | sqfstar -comp zstd -Xcompression-level 3 -b 131072 -no-duplicates "${OUT_SQFS}"
PIPE_STATUS=("${PIPESTATUS[@]}")
[ "${PIPE_STATUS[0]}" -eq 0 ] && [ "${PIPE_STATUS[1]}" -eq 0 ] \
    || fail "merge failed (stream=${PIPE_STATUS[0]} sqfstar=${PIPE_STATUS[1]}) — chunks preserved"
LOCAL_SIZE=$(stat -c%s "${OUT_SQFS}")
echo "STAGE 3 done in $((SECONDS - T0))s — $(numfmt --to=iec ${LOCAL_SIZE})"

# ── STAGE 4: verify merged image count ───────────────────────────────────────
ACTUAL=$(unsquashfs -l "${OUT_SQFS}" 2>/dev/null | grep -cE '\.(jpg|jpeg|png)$' || echo 0)
echo "Merged: ${ACTUAL}/${EXPECTED} images"
[ "${ACTUAL}" -eq "${EXPECTED}" ] || fail "image count mismatch ${ACTUAL}/${EXPECTED} — chunks preserved"

# ── STAGE 5: upload + verify remote size ─────────────────────────────────────
echo "=== STAGE 5: upload $(numfmt --to=iec ${LOCAL_SIZE}) $(date) ==="
T0=$SECONDS
s5cmd --endpoint-url "${ENDPOINT}" cp "${OUT_SQFS}" "${S3_DEST}" \
    || fail "s5cmd upload failed — local sqfs + chunks preserved"
UP_SECS=$((SECONDS - T0))
REMOTE_SIZE=$(s5cmd --endpoint-url "${ENDPOINT}" ls "${S3_DEST}" | awk '{print $3}')
echo "Uploaded in ${UP_SECS}s — local=${LOCAL_SIZE} remote=${REMOTE_SIZE}"
[ "${REMOTE_SIZE}" = "${LOCAL_SIZE}" ] || fail "remote size mismatch — local files preserved"

# ── STAGE 6: verified — delete local ─────────────────────────────────────────
rm -f "${OUT_SQFS}"
find "${TASK_DIR}" -name "chunk_*.sqfs" -delete
rmdir "${TASK_DIR}"/resume_* 2>/dev/null || true
echo "Local cleanup done."

echo "=== bq_download task=${TASK_ID} COMPLETE $(date) ==="
notify "bq_download task ${TASK_ID}/${NUM_JOBS}: done" \
    "${ACTUAL} images → ${S3_DEST} ($(numfmt --to=iec ${LOCAL_SIZE})), upload ${UP_SECS}s"
exit 0
