#!/bin/bash
#SBATCH --partition=unkillable          # Ask for unkillable job
#SBATCH --cpus-per-task=2               # Ask for 2 CPUs
#SBATCH --gres=gpu:1                    # Ask for 1 GPU
#SBATCH --mem=20G                       # Ask for 20 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Run the script
python end_to_end.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Vermont/' \
--image_folder '2022_05_13' \
--model_localize_type fasterrcnn_resnet50_fpn \
--model_localize '/home/mila/a/aditya.jain/logs/v1_localization_model_fasterrcnn_resnet50_fpn_2021-08-17-12-06.pt' \
--model_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v02_resnet50_2022-08-01-07-33.pt' \
--model_moth_type 'resnet50' \
--model_moth_nonmoth '/home/mila/a/aditya.jain/logs/moth-nonmoth-effv2b3_20220506_061527_30.pth' \
--model_moth_nonmoth_type 'tf_efficientnetv2_b3' \
--category_map_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth_category-map_4Aug2022.json' \
--category_map_moth_nonmoth '/home/mila/a/aditya.jain/logs/05-moth-nonmoth_category_map.json' \
--model_moth_cnn '/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v02_resnet50_2022-08-01-07-33.pt' \
--image_resize 300 \
--weight_cnn 1 \
--weight_iou 1 \
--weight_box_ratio 1 \
--weight_distance 1 \
--frame_rate 5 \
--scale_factor 0.4 \
--region 'Vermont'




