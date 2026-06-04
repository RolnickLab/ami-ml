#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=northamerica_fetch_webdataset
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/northamerica_fetch_webdataset_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

# Sharded fetch-and-pack for northamerica_butterflies.
# Replaces the old fetch + webdataset two-step for large datasets.
# Images are downloaded in chunks of CHUNK_SIZE, packed into webdataset tars,
# then deleted — keeping peak disk usage to ~chunk_size images at a time.
# The job is fully resumable: re-submit to continue from the last completed chunk.

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
DATASET="northamerica_butterflies"
SPLITS="$BASE_DIR/data/$DATASET/splits"
WBDS="/project/rrg-bengioy-ad/melabbas/$DATASET/webdataset"   # rrg storage (intentional)
TEMP_BASE="${SCRATCH:-/scratch/melabbas}/$DATASET/temp_images"

LABEL_COL="verbatimScientificName"
IMAGE_PATH_COL="image_path"
URL_COL="identifier"
RESIZE=450
CHUNK_SIZE=10000
NUM_WORKERS=32
MAX_SHARD_SIZE=$((100 * 1024 * 1024))   # 100 MB per shard

cd "$BASE_DIR"
source .venv/bin/activate

mkdir -p "$WBDS" "$TEMP_BASE"

echo "=== northamerica fetch-and-pack started at $(date) ==="

# --- Train split (saves category map for reuse by val/test) ---
ami-dataset fetch-and-pack \
  --annotations-csv "$SPLITS/train.csv" \
  --temp-dir "$TEMP_BASE/train" \
  --webdataset-dir "$WBDS/train" \
  --split train \
  --label-column "$LABEL_COL" \
  --image-path-column "$IMAGE_PATH_COL" \
  --url-column "$URL_COL" \
  --save-category-map-json "$WBDS/northamerica_butterflies_category_map.json" \
  --max-shard-size "$MAX_SHARD_SIZE" \
  --resize-min-size "$RESIZE" \
  --chunk-size "$CHUNK_SIZE" \
  --num-workers "$NUM_WORKERS"

echo "=== train done at $(date) ==="
notify "northamerica fetch-and-pack: train done" \
  "Train split packed. Starting val... ($(ls $WBDS/train/*.tar 2>/dev/null | wc -l) shards)"

# --- Val split ---
ami-dataset fetch-and-pack \
  --annotations-csv "$SPLITS/val.csv" \
  --temp-dir "$TEMP_BASE/val" \
  --webdataset-dir "$WBDS/val" \
  --split val \
  --label-column "$LABEL_COL" \
  --image-path-column "$IMAGE_PATH_COL" \
  --url-column "$URL_COL" \
  --category-map-json "$WBDS/northamerica_butterflies_category_map.json" \
  --max-shard-size "$MAX_SHARD_SIZE" \
  --resize-min-size "$RESIZE" \
  --chunk-size "$CHUNK_SIZE" \
  --num-workers "$NUM_WORKERS"

echo "=== val done at $(date) ==="

# --- Test split ---
ami-dataset fetch-and-pack \
  --annotations-csv "$SPLITS/test.csv" \
  --temp-dir "$TEMP_BASE/test" \
  --webdataset-dir "$WBDS/test" \
  --split test \
  --label-column "$LABEL_COL" \
  --image-path-column "$IMAGE_PATH_COL" \
  --url-column "$URL_COL" \
  --category-map-json "$WBDS/northamerica_butterflies_category_map.json" \
  --max-shard-size "$MAX_SHARD_SIZE" \
  --resize-min-size "$RESIZE" \
  --chunk-size "$CHUNK_SIZE" \
  --num-workers "$NUM_WORKERS"

echo "=== test done at $(date) ==="

# Summary
TRAIN_SHARDS=$(ls "$WBDS/train"/*.tar 2>/dev/null | wc -l)
VAL_SHARDS=$(ls "$WBDS/val"/*.tar 2>/dev/null | wc -l)
TEST_SHARDS=$(ls "$WBDS/test"/*.tar 2>/dev/null | wc -l)

echo "Shards: train=$TRAIN_SHARDS val=$VAL_SHARDS test=$TEST_SHARDS"
notify "northamerica fetch-and-pack done" \
  "All 3 splits complete. train=$TRAIN_SHARDS val=$VAL_SHARDS test=$TEST_SHARDS shards in $WBDS"
