#!/bin/bash

# Load absolute data paths 
set -o allexport
source .env
set +o allexport

# Calculate data statistics
datasets_count=$(ls $GLOBAL_MODEL_DATASET_PATH | wc -l)
num_images=$(find $GLOBAL_MODEL_DATASET_PATH -type f | wc -l)
dataset_size=$(du -sh $GLOBAL_MODEL_DATASET_PATH)

# Print statistics
echo "Number of dataset sources: $datasets_count"
echo "Number of images: $num_images"
echo "Dataset size: $dataset_size"