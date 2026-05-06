#!/bin/bash
#SBATCH --job-name=test_new_pipeline
#SBATCH --account=def-drolnick
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/project/6068129/melabbas/data/test_new_pipeline-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

BASE_DIR="/project/6068129/melabbas/ami-ml"
DWCA="$BASE_DIR/data/vermont_butterflies.zip"
OUT="/project/6068129/melabbas/data/test_new_pipeline"
SPLITS="$OUT/splits"
WBDS="$OUT/webdataset"
TEMP="$OUT/temp_images"

cd "$BASE_DIR"
source .venv/bin/activate

echo "================================================================"
echo "INPUT: DwCA inspection"
echo "================================================================"
python3 - <<PYEOF
from dwca.read import DwCAReader
dwca_path = '${DWCA}'
with DwCAReader(dwca_path) as dwca:
    media = dwca.pd_read('multimedia.txt', on_bad_lines='skip', low_memory=False)
    occ   = dwca.pd_read('occurrence.txt', on_bad_lines='skip', low_memory=False)
print(f'  multimedia rows (image URLs): {len(media):,}')
print(f'  occurrence rows:              {len(occ):,}')
print(f'  species (verbatimScientificName): {occ["verbatimScientificName"].nunique()}')
print(f'  lifeStage distribution:')
for k,v in occ['lifeStage'].value_counts(dropna=False).items():
    print(f'    {k}: {v:,}')
PYEOF

echo ""
echo "================================================================"
echo "STAGE 1: clean-dataset (directly on DwCA, no verify step)"
echo "================================================================"
if [[ -f "$SPLITS/clean.csv" ]]; then
    echo "  clean.csv already exists, skipping."
else
    ami-dataset clean-dataset \
      --dwca-file "$DWCA" \
      --output-csv "$SPLITS/clean.csv"
fi

echo ""
echo "--- clean.csv inspection ---"
python3 - <<PYEOF
import pandas as pd
df = pd.read_csv('$SPLITS/clean.csv')
print(f'  Rows (images):     {len(df):,}')
print(f'  Columns:           {list(df.columns)}')
print(f'  Species:           {df["verbatimScientificName"].nunique()}')
print(f'  Has identifier:    {"identifier" in df.columns}')
print(f'  Has image_path:    {"image_path" in df.columns}')
print(f'  lifeStage values:  {df["lifeStage"].value_counts(dropna=False).to_dict()}')
print(f'  Sample identifier: {df["identifier"].iloc[0]}')
print(f'  Sample image_path: {df["image_path"].iloc[0]}')
PYEOF

echo ""
echo "================================================================"
echo "STAGE 2: split-dataset"
echo "================================================================"
if [[ -f "$SPLITS/train.csv" ]]; then
    echo "  splits already exist, skipping."
else
    ami-dataset split-dataset \
      --dataset-csv "$SPLITS/clean.csv" \
      --split-prefix "$SPLITS" \
      --category-key verbatimScientificName \
      --max-instances 50 \
      --min-instances 5 \
      --test-frac 0.2 \
      --val-frac 0.2
fi

echo ""
echo "--- splits inspection ---"
python3 - <<PYEOF
import pandas as pd
for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'$SPLITS/{split}.csv')
    print(f'  {split}: {len(df):,} images, {df["verbatimScientificName"].nunique()} species')
    print(f'    has identifier: {"identifier" in df.columns}')
    print(f'    has image_path: {"image_path" in df.columns}')
    counts = df['verbatimScientificName'].value_counts()
    print(f'    images/species — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}')
PYEOF

echo ""
echo "================================================================"
echo "STAGE 3: fetch-and-pack (chunk=200 for speed)"
echo "================================================================"
mkdir -p "$WBDS/train" "$WBDS/val" "$WBDS/test" "$TEMP"

ami-dataset fetch-and-pack \
  --annotations-csv "$SPLITS/train.csv" \
  --temp-dir "$TEMP/train" \
  --webdataset-dir "$WBDS/train" \
  --split train \
  --label-column verbatimScientificName \
  --image-path-column image_path \
  --url-column identifier \
  --resize-min-size 450 \
  --save-category-map-json "$WBDS/category_map.json" \
  --chunk-size 200 \
  --num-workers 8

echo "  train done at $(date)"

ami-dataset fetch-and-pack \
  --annotations-csv "$SPLITS/val.csv" \
  --temp-dir "$TEMP/val" \
  --webdataset-dir "$WBDS/val" \
  --split val \
  --label-column verbatimScientificName \
  --image-path-column image_path \
  --url-column identifier \
  --resize-min-size 450 \
  --category-map-json "$WBDS/category_map.json" \
  --chunk-size 200 \
  --num-workers 8

echo "  val done at $(date)"

ami-dataset fetch-and-pack \
  --annotations-csv "$SPLITS/test.csv" \
  --temp-dir "$TEMP/test" \
  --webdataset-dir "$WBDS/test" \
  --split test \
  --label-column verbatimScientificName \
  --image-path-column image_path \
  --url-column identifier \
  --resize-min-size 450 \
  --category-map-json "$WBDS/category_map.json" \
  --chunk-size 200 \
  --num-workers 8

echo "  test done at $(date)"

echo ""
echo "================================================================"
echo "OUTPUT: webdataset inspection"
echo "================================================================"
python3 - <<PYEOF
import tarfile, glob, json

wbds_dir = '$WBDS'

with open(f'{wbds_dir}/category_map.json') as f:
    cat_map = json.load(f)
idx_to_label = {v: k for k, v in cat_map.items()}
print(f'  category_map.json: {len(cat_map)} classes')

for split in ['train', 'val', 'test']:
    shards = sorted(glob.glob(f'{wbds_dir}/{split}/*.tar'))
    total_images = 0
    class_counts = {}
    for shard in shards:
        with tarfile.open(shard) as tf:
            for m in tf.getmembers():
                if m.name.endswith('.cls'):
                    cls_idx = int(tf.extractfile(m).read().strip())
                    label = idx_to_label.get(cls_idx, str(cls_idx))
                    class_counts[label] = class_counts.get(label, 0) + 1
                    total_images += 1
    counts = list(class_counts.values())
    print(f'  {split}: {len(shards)} shards, {total_images} images, {len(class_counts)} species')
    if counts:
        print(f'    images/species — min: {min(counts)}, max: {max(counts)}, mean: {sum(counts)/len(counts):.1f}')

remaining = glob.glob('$TEMP/**/*.jpg', recursive=True)
print(f'  temp images remaining on disk: {len(remaining)} (should be 0)')
PYEOF

echo ""
echo "--- sample shard contents (first train shard) ---"
FIRST_SHARD=$(ls "$WBDS/train/"*.tar | head -1)
echo "  $(basename $FIRST_SHARD):"
tar -tf "$FIRST_SHARD" | head -10

echo ""
echo "Pipeline test complete at $(date)"
~/bin/notify "test_new_pipeline done" "End-to-end pipeline test complete. Results in $OUT"
