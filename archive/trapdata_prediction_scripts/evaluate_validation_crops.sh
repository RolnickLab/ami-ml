#! /bin/bash

set -o nounset
set -o errexit

###
# Usage:
# export AMIDATA=/home/username/Projects/AMI/data
# cd mothAI/trapdata_prediction_scripts/
# bash evaluate_validation_crops.sh
# python merge_validation_crops_results.py
###

python classification.py \
    --data_dir "${AMIDATA}/validation_crops/" \
    --image_folder "Quebec" \
    --model_moth "${AMIDATA}/models/quebec-vermont-moth-model_v02_efficientnetv2-b3_2022-09-08-15-44.pt" \
    --model_moth_type "tf_efficientnetv2_b3" \
    --category_map_moth "${AMIDATA}/models/quebec-vermont-moth_category-map_4Aug2022.json"

python classification.py \
    --data_dir "${AMIDATA}/validation_crops/" \
    --image_folder "Quebec" \
    --model_moth "${AMIDATA}/models/quebec-vermont-moth-model_v07_resnet50_2022-12-22-07-54.pt" \
    --model_moth_type "resnet50" \
    --category_map_moth "${AMIDATA}/models/quebec-vermont_moth-category-map_19Jan2023.json"
