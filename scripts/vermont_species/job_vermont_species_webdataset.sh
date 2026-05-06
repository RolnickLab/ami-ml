#!/bin/bash
#SBATCH --account=def-drolnick
#SBATCH --job-name=vermont_species_webdataset
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=vermont_species_webdataset_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

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
  --webdataset-pattern "$WBDS/vermont_butterflies_train_verbatim-%06d.tar" \
  --image-path-column image_path \
  --label-column verbatimScientificName \
  --resize-min-size 450 \
  --save-category-map-json "$WBDS/vermont_butterflies_category_map.json"

ami-dataset create-webdataset \
  --annotations-csv "$SPLITS/val.csv" \
  --dataset-path "$IMAGES" \
  --webdataset-pattern "$WBDS/vermont_butterflies_val_verbatim-%06d.tar" \
  --image-path-column image_path \
  --label-column verbatimScientificName \
  --resize-min-size 450 \
  --category-map-json "$WBDS/vermont_butterflies_category_map.json"

ami-dataset create-webdataset \
  --annotations-csv "$SPLITS/test.csv" \
  --dataset-path "$IMAGES" \
  --webdataset-pattern "$WBDS/vermont_butterflies_test_verbatim-%06d.tar" \
  --image-path-column image_path \
  --label-column verbatimScientificName \
  --resize-min-size 450 \
  --category-map-json "$WBDS/vermont_butterflies_category_map.json"

echo "Webdataset completed at $(date)"
