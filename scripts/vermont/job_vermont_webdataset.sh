#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_webdataset
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=vermont_webdataset_%j.out

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"
IMAGES="$BASE_DIR/data/vermont_butterflies/images"
SPLITS="$BASE_DIR/data/vermont_butterflies/splits"
WBDS="$BASE_DIR/data/vermont_butterflies/webdataset"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p "$WBDS"

ami-dataset create-webdataset \
  --annotations-csv "$SPLITS/train.csv" \
  --dataset-path "$IMAGES" \
  --webdataset-pattern "$WBDS/vermont_train_verbatim-%06d.tar" \
  --image-path-column image_path \
  --label-column verbatimScientificName \
  --resize-min-size 450 \
  --save-category-map-json "$WBDS/vermont_category_map_verbatim.json"

ami-dataset create-webdataset \
  --annotations-csv "$SPLITS/val.csv" \
  --dataset-path "$IMAGES" \
  --webdataset-pattern "$WBDS/vermont_val_verbatim-%06d.tar" \
  --image-path-column image_path \
  --label-column verbatimScientificName \
  --resize-min-size 450 \
  --category-map-json "$WBDS/vermont_category_map_verbatim.json"

ami-dataset create-webdataset \
  --annotations-csv "$SPLITS/test.csv" \
  --dataset-path "$IMAGES" \
  --webdataset-pattern "$WBDS/vermont_test_verbatim-%06d.tar" \
  --image-path-column image_path \
  --label-column verbatimScientificName \
  --resize-min-size 450 \
  --category-map-json "$WBDS/vermont_category_map_verbatim.json"

echo "Webdataset completed at $(date)"
