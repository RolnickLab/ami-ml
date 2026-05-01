#!/bin/bash
# Build global WebDataset from all 10 sqfs files.
#
# Requires a BQ-exported CSV at CSV_PATH with columns:
#   photo_id, relative_local_path, species_name (or class_id), + any other metadata
#
# Strategy: two batches of BATCH_SIZE sqfs to stay under 7 TB NVMe peak:
#   Batch 1 (sqfs 0-4): --tar-mode w  (create new tars)
#   Batch 2 (sqfs 5-9): --tar-mode a  (append, idempotent on retry)
#
# Export the CSV from BQ before submitting:
#   bq query --format=csv --max_rows=15000000 \
#     "SELECT ti.photo_id, ti.relative_local_path, ti.dataset_source_uuid,
#             tx.species_name, tx.inat_taxon_id, tx.family, tx.gbif_accepted_taxon_key
#      FROM leps-ai.global_butterflies_2604.training_images ti
#      JOIN leps-ai.global_butterflies_2604.inat_taxa tx USING (inat_taxon_id)
#      JOIN (SELECT DISTINCT dataset_source_uuid
#            FROM leps-ai.global_butterflies_2604.training_images_downloads
#            WHERE fetch_status='downloaded') d USING (dataset_source_uuid)
#      WHERE tx.species_name IS NOT NULL" > /project/6068129/melabbas/ami-ml/data/global_wds_export.csv
#
#SBATCH --account=def-drolnick
#SBATCH --job-name=build_wds_global
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --tmp=7000G
#SBATCH --time=12:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/build_wds_global_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com
#SBATCH --exclude=fc30554

NVME="${SLURM_TMPDIR}"
OUTPUT_DIR="/scratch/melabbas/global_wds"
SPLITS_DIR="/project/6068129/melabbas/ami-ml/data/splits"   # contains train.csv val.csv test.csv
IMAGES_PER_SHARD=1000
BATCH_SIZE=5
PACK_WORKERS=16

echo "=== build_wds_global started at $(date) ==="
echo "Node: $(hostname)"
echo "NVMe: ${NVME}  ($(df -h ${NVME} | tail -1 | awk '{print $4}') free)"
echo "CSV:  ${CSV_PATH}  ($(du -sh ${CSV_PATH} | cut -f1))"
echo ""

for SPLIT in train val test; do
    if [ ! -f "${SPLITS_DIR}/${SPLIT}.csv" ]; then
        echo "ERROR: ${SPLITS_DIR}/${SPLIT}.csv not found"
        echo "Run split_csv.py first to generate the split CSVs."
        exit 1
    fi
done
echo "Split CSVs: $(wc -l < ${SPLITS_DIR}/train.csv) train  $(wc -l < ${SPLITS_DIR}/val.csv) val  $(wc -l < ${SPLITS_DIR}/test.csv) test rows"
echo ""

# ── Locate all sqfs files ─────────────────────────────────────────────────────
find_sqfs() {
    local T=$1
    for DIR in /project/rrg-bengioy-ad/melabbas /project/6068129/melabbas /scratch/melabbas; do
        [ -f "${DIR}/task_${T}.sqfs" ] && echo "${DIR}/task_${T}.sqfs" && return 0
    done
    echo ""
}

SQFS_PATHS=()
echo "Locating sqfs files..."
for T in $(seq 0 9); do
    P=$(find_sqfs $T)
    if [ -z "$P" ]; then
        echo "ERROR: could not find task_${T}.sqfs"
        exit 1
    fi
    SQFS_PATHS+=("$P")
    echo "  task_${T}: ${P}  ($(du -sh ${P} | cut -f1))"
done
echo ""

# ── Setup Lustre output with HDD stripe ──────────────────────────────────────
for SPLIT in train val test; do
    mkdir -p "${OUTPUT_DIR}/${SPLIT}"
    lfs setstripe -c -1 -S 4m -p ddn_hdd "${OUTPUT_DIR}/${SPLIT}" 2>/dev/null || true
done
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

# ── Python environment ────────────────────────────────────────────────────────
cd /project/6068129/melabbas/ami-ml
module load StdEnv/2023 arrow/17.0.0
source .venv/bin/activate

# ── Batch 1: sqfs 0–(BATCH_SIZE-1), create new tars ──────────────────────────
BATCH1_END=$((BATCH_SIZE - 1))
echo "=== BATCH 1: sqfs 0–${BATCH1_END}  (mode=w) at $(date) ==="

python src/dataset_tools/bq_squashfs/create_webdataset.py \
    --split-csvs     train:${SPLITS_DIR}/train.csv val:${SPLITS_DIR}/val.csv test:${SPLITS_DIR}/test.csv \
    --sqfs-paths     "${SQFS_PATHS[@]:0:${BATCH_SIZE}}" \
    --sqfs-start-idx 0 \
    --images-per-shard ${IMAGES_PER_SHARD} \
    --nvme-dir       "${NVME}" \
    --output-dir     "${OUTPUT_DIR}" \
    --pack-workers   ${PACK_WORKERS} \
    --tar-mode       w

BATCH1_EXIT=$?
echo ""
if [ ${BATCH1_EXIT} -ne 0 ]; then
    echo "ERROR: batch 1 failed (exit=${BATCH1_EXIT})"
    notify "build_wds_global: FAILED (batch 1)" \
        "exit=${BATCH1_EXIT} — check build_wds_global_${SLURM_JOB_ID}.out"
    exit ${BATCH1_EXIT}
fi

# ── Batch 2: sqfs BATCH_SIZE–9, append to existing tars ──────────────────────
echo "=== BATCH 2: sqfs ${BATCH_SIZE}–9  (mode=a) at $(date) ==="

python src/dataset_tools/bq_squashfs/create_webdataset.py \
    --split-csvs     train:${SPLITS_DIR}/train.csv val:${SPLITS_DIR}/val.csv test:${SPLITS_DIR}/test.csv \
    --sqfs-paths     "${SQFS_PATHS[@]:${BATCH_SIZE}}" \
    --sqfs-start-idx ${BATCH_SIZE} \
    --images-per-shard ${IMAGES_PER_SHARD} \
    --nvme-dir       "${NVME}" \
    --output-dir     "${OUTPUT_DIR}" \
    --pack-workers   ${PACK_WORKERS} \
    --tar-mode       a

BATCH2_EXIT=$?
echo ""
echo "=== build_wds_global done at $(date) (exit=${BATCH2_EXIT}) ==="

if [ ${BATCH2_EXIT} -eq 0 ]; then
    for SPLIT in train val test; do
        COUNT=$(ls "${OUTPUT_DIR}/${SPLIT}/"*.tar 2>/dev/null | wc -l)
        SIZE=$(du -sh "${OUTPUT_DIR}/${SPLIT}" 2>/dev/null | cut -f1)
        echo "  ${SPLIT}: ${COUNT} shards  ${SIZE}"
    done
    notify "build_wds_global: done" \
        "exit=0  job=${SLURM_JOB_ID} — train/val/test shards written to ${OUTPUT_DIR}"
else
    notify "build_wds_global: FAILED (batch 2)" \
        "exit=${BATCH2_EXIT} — check build_wds_global_${SLURM_JOB_ID}.out"
fi

exit ${BATCH2_EXIT}
