#!/bin/bash
#SBATCH --partition=unkillable                # Ask for main job
#SBATCH --cpus-per-task=4                     # Ask for 1 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM

## this bash script archives the moth data into one file

start=`date +%s`

for i in 1
do
	tar -cf /home/mila/a/aditya.jain/scratch/GBIF_Data/uk-moth-data_archived.tar /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk
	tar -cf /home/mila/a/aditya.jain/scratch/GBIF_Data/non-moth-data_archived.tar /home/mila/a/aditya.jain/scratch/GBIF_Data/nonmoths
done


end=`date +%s`

runtime=$((end-start))
echo 'Time taken for archiving the data in seconds' $runtime

