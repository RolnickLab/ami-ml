#!/bin/bash

python3 /home/mila/l/leonard.pasi/ami-ml/active_learning/src/score_gbif_images.py \
    --image_list_csv /home/mila/l/leonard.pasi/scratch/data/classification/quebec-vermont_val-split.csv \
    --image_dir /network/scratch/a/aditya.jain/GBIF_Data/moths_world/ \
    --save_path /home/mila/l/leonard.pasi/scratch/data/classification/scores \
    --save_name "quebec_vermont_pool_scores.csv" \
    --image_resize 300 \
    --ckpt_path /home/mila/l/leonard.pasi/scratch/models/classification/model2/quebec-vermont-moth-model_82x20mlb.pt \
    --ckpt_path /home/mila/l/leonard.pasi/scratch/models/classification/model2/quebec-vermont-moth-model_ih6cxh1t.pt \
    --num_classes 3150 \
    --scoring_func entropy \
    --scoring_func mutual_info \
    --scoring_func least_confidence \
    --scoring_func margin_sampling \
    --scoring_func variation_ratios
