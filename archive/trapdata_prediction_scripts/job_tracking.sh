#!/bin/bash
#SBATCH --partition=main                # Ask for main job
#SBATCH --cpus-per-task=4               # Ask for 4 CPUs
#SBATCH --gres=gpu:1                    # Ask for 1 GPU
#SBATCH --mem=4G                        # Ask for 4 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Run the script
python tracks_w_classification_multiple.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Vermont/' \
--image_folder '2022_05_13' \
--model_moth_cnn '/home/mila/a/aditya.jain/logs/quebec-vermont-moth-model_v02_resnet50_2022-08-01-07-33.pt' \
--category_map_moth '/home/mila/a/aditya.jain/logs/quebec-vermont-moth_category-map_4Aug2022.json' \
--image_resize 300 \
--weight_cnn 1 \
--weight_iou 1 \
--weight_box_ratio 1 \
--weight_distance 1 




