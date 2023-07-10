#!/bin/bash
#SBATCH --partition=long-cpu        # Ask for long cpu job
#SBATCH --cpus-per-task=3           # Ask for 3 CPUs
#SBATCH --mem=650G                  # Ask for 700 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

start=`date +%s`
# 4. Launch your script
python 02-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_world/ \
--dwca_file /home/mila/a/aditya.jain/scratch/GBIF_Data/leps_images_adult-imago.zip \
--species_checklist /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_25Apr2023.csv \
--max_images_per_species 1000 \
--resume_session True 
    
end=`date +%s`
runtime=$((end-start))
echo 'Time taken for downloading the data in seconds' $runtime