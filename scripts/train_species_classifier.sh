#!/bin/bash
#
# Single-script species classifier training pipeline.
# Orchestrates the full flow from a GBIF Darwin Core Archive (DwC-A)
# to a trained ConvNeXt-Tiny species classification model.
#
# Usage:
#   bash scripts/train_species_classifier.sh
#
# Prerequisites:
#   - uv installed and project dependencies synced (uv sync --extra dev)
#   - DwC-A zip file at DWCA_FILE path below
#

set -euo pipefail

# ============================================================
# Configuration — edit these paths as needed
# ============================================================
DWCA_FILE="0007113-260208012135463.zip"
DATASET_PATH="dataset-out"
OUTPUT_DIR="output/species_classifier"
LABEL_COLUMN="verbatimScientificName"
MIN_INSTANCES=3
# MIN_INSTANCES=0  # for tiny datasets (<200 images) where few species meet the threshold
NUM_WORKERS=8

# Training hyperparameters
MODEL_TYPE="convnext_tiny_in22k"
TOTAL_EPOCHS=35
WARMUP_EPOCHS=3
EARLY_STOPPING=5
LOSS_FUNCTION="cross_entropy"
LABEL_SMOOTHING=0.1
LR_SCHEDULER="cosine"
LEARNING_RATE=0.001
BATCH_SIZE=64
IMAGE_INPUT_SIZE=128

# Webdataset settings
RESIZE_MIN_SIZE=450
MAX_SHARD_SIZE=$((100 * 1024 * 1024))  # 100 MB

# Weights & Biases (optional — leave empty to disable)
WANDB_ENTITY=""
WANDB_PROJECT=""
WANDB_RUN_NAME=""

# ============================================================
# Derived paths (generally don't need to edit)
# ============================================================
VERIFIED_CSV="${OUTPUT_DIR}/verified_images.csv"
CLEAN_CSV="${VERIFIED_CSV%.csv}_clean.csv"
AUGMENTED_CSV="${OUTPUT_DIR}/annotations_with_species.csv"
CATEGORY_MAP="${OUTPUT_DIR}/category_map.json"
SPLIT_PREFIX="${OUTPUT_DIR}/split"
TRAIN_CSV="${SPLIT_PREFIX}/train.csv"
VAL_CSV="${SPLIT_PREFIX}/val.csv"
TEST_CSV="${SPLIT_PREFIX}/test.csv"
TRAIN_WBDS_DIR="${OUTPUT_DIR}/webdataset_train"
VAL_WBDS_DIR="${OUTPUT_DIR}/webdataset_val"
TEST_WBDS_DIR="${OUTPUT_DIR}/webdataset_test"
MODEL_SAVE_DIR="${OUTPUT_DIR}/model"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# Helper functions
# ============================================================
step_header() {
    echo ""
    echo "============================================================"
    echo "STEP $1: $2"
    echo "============================================================"
    echo ""
}

count_tar_files() {
    # Count .tar files in a directory
    local dir="$1"
    find "$dir" -maxdepth 1 -name "*.tar" | wc -l
}

build_shard_pattern() {
    # Build a brace-expansion webdataset pattern from tar files in a directory
    # e.g. "output/webdataset_train/shard-{000000..000003}.tar"
    local dir="$1"
    local prefix="$2"
    local count
    count=$(count_tar_files "$dir")
    if [ "$count" -eq 0 ]; then
        echo "ERROR: No tar files found in $dir" >&2
        return 1
    fi
    local last
    last=$(printf "%06d" $((count - 1)))
    echo "${dir}/${prefix}-{000000..${last}}.tar"
}

# ============================================================
# Ensure output directories exist
# ============================================================
mkdir -p "$DATASET_PATH" "$OUTPUT_DIR" "$SPLIT_PREFIX" "$TRAIN_WBDS_DIR" "$VAL_WBDS_DIR" "$TEST_WBDS_DIR" "$MODEL_SAVE_DIR"

# Keep track of total time
TOTAL_SECONDS=$SECONDS

# ============================================================
# Step 1: Fetch images from DwC-A
# ============================================================
if [ -d "$DATASET_PATH" ] && [ "$(find "$DATASET_PATH" -name '*.jpg' 2>/dev/null | head -1)" ]; then
    echo "SKIP Step 1: Images already exist in $DATASET_PATH"
else
    step_header 1 "Fetch images from DwC-A"
    uv run ami-dataset fetch-images \
        --dataset-path "$DATASET_PATH" \
        --dwca-file "$DWCA_FILE" \
        --num-workers "$NUM_WORKERS"

    echo "Step 1 complete. Time elapsed: $((SECONDS - TOTAL_SECONDS))s"
fi

# ============================================================
# Step 2: Verify downloaded images
# ============================================================
if [ -f "$VERIFIED_CSV" ]; then
    echo "SKIP Step 2: Verified CSV already exists at $VERIFIED_CSV"
else
    step_header 2 "Verify downloaded images"
    uv run ami-dataset verify-images \
        --dataset-path "$DATASET_PATH" \
        --dwca-file "$DWCA_FILE" \
        --results-csv "$VERIFIED_CSV" \
        --num-workers "$NUM_WORKERS"

    echo "Step 2 complete. Time elapsed: $((SECONDS - TOTAL_SECONDS))s"
fi

# ============================================================
# Step 3: Clean dataset (filter thumbnails, duplicates, etc.)
# ============================================================
if [ -f "$CLEAN_CSV" ]; then
    echo "SKIP Step 3: Clean CSV already exists at $CLEAN_CSV"
else
    step_header 3 "Clean dataset"
    uv run ami-dataset clean-dataset \
        --dwca-file "$DWCA_FILE" \
        --verified-data-csv "$VERIFIED_CSV" \
        --remove-non-adults false

    echo "Step 3 complete. Time elapsed: $((SECONDS - TOTAL_SECONDS))s"
fi

# ============================================================
# Step 4: Augment CSV with species names from DwC-A
# ============================================================
if [ -f "$AUGMENTED_CSV" ] && [ -f "$CATEGORY_MAP" ]; then
    echo "SKIP Step 4: Augmented CSV and category map already exist"
else
    step_header 4 "Build species list and augment annotations"
    uv run python "$SCRIPT_DIR/build_species_list.py" \
        --dwca-file "$DWCA_FILE" \
        --annotations-csv "$CLEAN_CSV" \
        --output-csv "$AUGMENTED_CSV" \
        --category-map-json "$CATEGORY_MAP" \
        --label-column "$LABEL_COLUMN"

    echo "Step 4 complete. Time elapsed: $((SECONDS - TOTAL_SECONDS))s"
fi

# ============================================================
# Step 5: Split dataset (stratified train/val/test)
# ============================================================
if [ -f "$TRAIN_CSV" ] && [ -f "$VAL_CSV" ] && [ -f "$TEST_CSV" ]; then
    echo "SKIP Step 5: Split CSVs already exist"
else
    step_header 5 "Split dataset into train/val/test"
    uv run ami-dataset split-dataset \
        --dataset-csv "$AUGMENTED_CSV" \
        --split-prefix "$SPLIT_PREFIX" \
        --category-key "$LABEL_COLUMN" \
        --max-instances -1 \
        --min-instances "$MIN_INSTANCES"
    # For tiny datasets (<200 images): add these flags to lower the per-species threshold
    # --val-frac 0.3 --test-frac 0.2

    echo "Step 5 complete. Time elapsed: $((SECONDS - TOTAL_SECONDS))s"
fi

# ============================================================
# Step 6: Create webdatasets (train, val, test)
# ============================================================
WBDS_ARGS=(
    --dataset-path "$DATASET_PATH"
    --image-path-column "image_path"
    --label-column "$LABEL_COLUMN"
    --category-map-json "$CATEGORY_MAP"
    --resize-min-size "$RESIZE_MIN_SIZE"
    --max-shard-size "$MAX_SHARD_SIZE"
)

# 6a: Training webdataset
if [ "$(count_tar_files "$TRAIN_WBDS_DIR")" -gt 0 ]; then
    echo "SKIP Step 6a: Training webdataset shards already exist"
else
    step_header "6a" "Create training webdataset"
    uv run ami-dataset create-webdataset \
        --annotations-csv "$TRAIN_CSV" \
        --webdataset-pattern "${TRAIN_WBDS_DIR}/shard-%06d.tar" \
        "${WBDS_ARGS[@]}"

    echo "  Created $(count_tar_files "$TRAIN_WBDS_DIR") training shards"
fi

# 6b: Validation webdataset
if [ "$(count_tar_files "$VAL_WBDS_DIR")" -gt 0 ]; then
    echo "SKIP Step 6b: Validation webdataset shards already exist"
else
    step_header "6b" "Create validation webdataset"
    uv run ami-dataset create-webdataset \
        --annotations-csv "$VAL_CSV" \
        --webdataset-pattern "${VAL_WBDS_DIR}/shard-%06d.tar" \
        "${WBDS_ARGS[@]}"

    echo "  Created $(count_tar_files "$VAL_WBDS_DIR") validation shards"
fi

# 6c: Test webdataset
if [ "$(count_tar_files "$TEST_WBDS_DIR")" -gt 0 ]; then
    echo "SKIP Step 6c: Test webdataset shards already exist"
else
    step_header "6c" "Create test webdataset"
    uv run ami-dataset create-webdataset \
        --annotations-csv "$TEST_CSV" \
        --webdataset-pattern "${TEST_WBDS_DIR}/shard-%06d.tar" \
        "${WBDS_ARGS[@]}"

    echo "  Created $(count_tar_files "$TEST_WBDS_DIR") test shards"
fi

echo "Step 6 complete. Time elapsed: $((SECONDS - TOTAL_SECONDS))s"

# ============================================================
# Step 7: Train the species classifier
# ============================================================
step_header 7 "Train species classifier"

# Compute num_classes from category map
NUM_CLASSES=$(uv run python -c "import json; print(len(json.load(open('${CATEGORY_MAP}'))))")
echo "Number of classes: $NUM_CLASSES"

# Build webdataset shard patterns
TRAIN_PATTERN=$(build_shard_pattern "$TRAIN_WBDS_DIR" "shard")
VAL_PATTERN=$(build_shard_pattern "$VAL_WBDS_DIR" "shard")
TEST_PATTERN=$(build_shard_pattern "$TEST_WBDS_DIR" "shard")

echo "Train pattern: $TRAIN_PATTERN"
echo "Val pattern:   $VAL_PATTERN"
echo "Test pattern:  $TEST_PATTERN"

# Build optional wandb args
WANDB_ARGS=()
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT")
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

uv run ami-classification train-model \
    --model_type "$MODEL_TYPE" \
    --num_classes "$NUM_CLASSES" \
    --total_epochs "$TOTAL_EPOCHS" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --early_stopping "$EARLY_STOPPING" \
    --train_webdataset "$TRAIN_PATTERN" \
    --val_webdataset "$VAL_PATTERN" \
    --test_webdataset "$TEST_PATTERN" \
    --image_input_size "$IMAGE_INPUT_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --learning_rate_scheduler "$LR_SCHEDULER" \
    --loss_function_type "$LOSS_FUNCTION" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --model_save_directory "$MODEL_SAVE_DIR" \
    "${WANDB_ARGS[@]}"

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Total time: $(( (SECONDS - TOTAL_SECONDS) / 60 )) minutes"
echo ""
echo "Outputs:"
echo "  Category map:    $CATEGORY_MAP"
echo "  Train CSV:       $TRAIN_CSV"
echo "  Val CSV:         $VAL_CSV"
echo "  Test CSV:        $TEST_CSV"
echo "  Train shards:    $TRAIN_WBDS_DIR/ ($(count_tar_files "$TRAIN_WBDS_DIR") files)"
echo "  Val shards:      $VAL_WBDS_DIR/ ($(count_tar_files "$VAL_WBDS_DIR") files)"
echo "  Test shards:     $TEST_WBDS_DIR/ ($(count_tar_files "$TEST_WBDS_DIR") files)"
echo "  Model:           $MODEL_SAVE_DIR/"
echo "============================================================"
