#!/bin/bash
#SBATCH --job-name=centralamerica_resnet50_baseline_run1
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --output=centralamerica_resnet50_baseline_run1.out

## Run this from the projet root directory.
## ./research/<sub_dir(s)>/<filename>.sh

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

# 4. Run python file
python model_prediction_on_trap_data.py \
--data_dir $ECCV2024_DATA \
--binary_model moth-nonmoth_resnet50_20230604_065440_30.pth \
--binary_model_type resnet50 \
--moth_model centralamerica_resnet50_baseline_run1.pth \
--moth_model_type resnet50 \
--category_map_binary_model "05-moth-nonmoth_category_map.json" \
--category_map_moth_model "03_moths_centralAmerica_category_map.json" \
--region CentralAmerica \
--global_species_list $SPECIES_LISTS_DIR/quebec-vermont-uk-denmark-panama_checklist_20231124.csv
