#!/bin/bash
#SBATCH --job-name=fgrained_model_evaluation_%j
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --output=fgrained_model_evaluation_%j.out

## Run this from the projet root directory.
## ./research/<sub_dir(s)>/<filename>.sh

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate ami-ml

# 3. Load the environment variables outside of python script
set -o allexport
source .env
set +o allexport

# 4. Run python file
python eccv2024/model_evaluation/fgrained_model_evaluation.py \
--run-name ne-america_resnet50_abla_imagenetamigbif_10ep_run3 \
--artifact moth-ai/ami-gbif-fine-grained/model:v75 \
--region NorthEasternAmerica \
--model-type resnet50 \
--model-dir $ECCV2024_DATA/models/fine_grained \
--category-map 01_ami-gbif_fine-grained_ne-america_category_map.json \
--insect-crops-dir $ECCV2024_DATA/camera_ready_amitraps/insect_crops \
--sp-exclusion-list-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle \
--ami-traps-taxonomy-map-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_taxonomy_map.csv \
--gbif-taxonomy-hierarchy-file $ECCV2024_DATA/camera_ready_amitraps/metadata/gbif_taxonomy_hierarchy.json


# python fgrained_model_evaluation.py \
# --run-name c-america_resnet50_abla_imagenetamigbif_10ep_run1 \
# --artifact moth-ai/ami-gbif-fine-grained/model:v74 \
# --region CentralAmerica \
# --model-type resnet50 \
# --model-dir $ECCV2024_DATA/models/fine_grained \
# --category-map "03_moths_centralAmerica_category_map.json" \
# --insect-crops-dir $ECCV2024_DATA/camera_ready_amitraps/insect_crops \
# --sp-exclusion-list-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle \
# --ami-traps-taxonomy-map-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_taxonomy_map.csv \
# --gbif-taxonomy-hierarchy-file $ECCV2024_DATA/camera_ready_amitraps/metadata/gbif_taxonomy_hierarchy.json


# python fgrained_model_evaluation.py \
# --run-name w-europe_resnet50_abla_imagenetamigbif_10ep_run1 \
# --artifact moth-ai/ami-gbif-fine-grained/model:v74 \
# --region WesternEurope \
# --model-type resnet50 \
# --model-dir $ECCV2024_DATA/models/fine_grained \
# --category-map "02_moths_westernEurope_category_map.json" \
# --insect-crops-dir $ECCV2024_DATA/camera_ready_amitraps/insect_crops \
# --sp-exclusion-list-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_sp_missing_in_ami-gbif.pickle \
# --ami-traps-taxonomy-map-file $ECCV2024_DATA/camera_ready_amitraps/metadata/ami-traps_taxonomy_map.csv \
# --gbif-taxonomy-hierarchy-file $ECCV2024_DATA/camera_ready_amitraps/metadata/gbif_taxonomy_hierarchy.json
