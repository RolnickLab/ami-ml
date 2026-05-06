#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=test_fetch_and_pack
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/project/6068129/melabbas/ami-ml/scripts/test_fetch_and_pack_%j.out

# Smoke test for fetch-and-pack: 30 rows, chunk_size=10 (3 chunks), 4 workers.
# Verifies: images fetched, shards created, temp dir cleaned, progress file deleted.

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
TEST_CSV="$BASE_DIR/data/test_fap_30rows.csv"
TEMP_DIR="${SCRATCH:-/scratch/melabbas}/test_fetch_and_pack/temp"
OUT_DIR="${SCRATCH:-/scratch/melabbas}/test_fetch_and_pack/webdataset"

cd "$BASE_DIR"
source .venv/bin/activate

mkdir -p "$OUT_DIR" "$TEMP_DIR"

echo "=== fetch-and-pack smoke test started at $(date) ==="

ami-dataset fetch-and-pack \
  --annotations-csv "$TEST_CSV" \
  --temp-dir "$TEMP_DIR" \
  --webdataset-dir "$OUT_DIR" \
  --split test_fap \
  --label-column verbatimScientificName \
  --image-path-column image_path \
  --url-column identifier \
  --chunk-size 10 \
  --num-workers 4 \
  --max-shard-size $((10 * 1024 * 1024)) \
  --shuffle-images False \
  --save-category-map-json "$OUT_DIR/test_fap_category_map.json"

echo ""
echo "=== Results ==="
echo "Shards produced:"
ls -lh "$OUT_DIR"/*.tar 2>/dev/null || echo "  (none)"

echo ""
echo "Temp dir contents (should be empty after success):"
find "$TEMP_DIR" -type f 2>/dev/null | head -10 || echo "  (empty)"

echo ""
echo "Progress file (should be gone after success):"
ls "$OUT_DIR"/test_fap_progress.json 2>/dev/null && echo "  EXISTS (unexpected)" || echo "  absent (correct)"

echo ""
echo "Category map:"
cat "$OUT_DIR/test_fap_category_map.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  {len(d)} categories')" || echo "  (not found)"

echo ""
echo "Shard integrity check:"
python3 - <<PYEOF
import glob, sys
try:
    import webdataset as wds
    shards = sorted(glob.glob("${OUT_DIR}/*.tar"))
    print(f"  {len(shards)} shards: {[s.split('/')[-1] for s in shards]}")
    count = 0
    dataset = wds.WebDataset("{" + ",".join(shards) + "}") if len(shards) > 1 else wds.WebDataset(shards[0])
    for sample in dataset:
        count += 1
    print(f"  {count} samples readable across all shards — OK")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "=== smoke test finished at $(date) ==="
notify "fetch-and-pack smoke test done" "$(ls "$OUT_DIR"/*.tar 2>/dev/null | wc -l) shards, check job output for integrity results"
