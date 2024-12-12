#!/bin/bash
#SBATCH --partition=long-cpu        # Ask for long cpu job
#SBATCH --cpus-per-task=2           # Ask for 4 CPUs
#SBATCH --mem=400G                  # Ask for 650 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

start=`date +%s`
# 4. Launch your script
python 02-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--dwca_file /home/mila/a/aditya.jain/scratch/GBIF_Data/leps_images_adult-imago.zip \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_24Aug2023.csv \
--max_images_per_species 1000 \
--num_workers 2
    
end=`date +%s`
runtime=$((end-start))
echo 'Time taken for downloading the data in seconds' $runtime
