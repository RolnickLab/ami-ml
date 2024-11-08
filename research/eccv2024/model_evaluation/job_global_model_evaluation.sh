#!/bin/bash
#SBATCH --job-name=eval_global_model_ami_traps
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=10G
#SBATCH --partition=unkillable            # Ask for unkillable job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2                 # Ask for 2 CPUs
#SBATCH --output=eval_global_model_ami_traps%j.out

## Run this from the projet root directory.
## ./research/<sub_dir(s)>/<filename>.sh

# 1. Load the required modules
module load miniconda/3

# 2. Load your environment
conda activate ami-ml

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

# 4. Run python file
python eccv2024/model_evaluation/fgrained_model_evaluation.py \
--run-name global_model_config23_21ep_timm_resnet50.pt \
--model-type timm_resnet50 \
--model-dir $GLOBAL_MODEL_DIR \
--sp-exclusion-list-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle \
--ami-traps-taxonomy-map-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_taxonomy_map.csv \
--category-map category_map.json \
--insect-crops-dir $ECCV2024_DATA/camera_ready_amitraps/insect_crops \
--gbif-taxonomy-hierarchy-file $GLOBAL_MODEL_DIR/gbif_taxonomy_hierarchy.json



