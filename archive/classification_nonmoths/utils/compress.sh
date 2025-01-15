#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs

## this bash script archives the moth and non data into tar files

start=`date +%s`

do 
	tar -cf moth_data-archived.tar /home/mila/a/aditya.jain/scratch/GBIF_Data/moths/
	tar -cf nonmoth_data-archived.tar /home/mila/a/aditya.jain/scratch/GBIF_Data/nonmoths/
done

end=`date +%s`

runtime=$((end-start))
echo 'Time taken for archiving the data in seconds' $runtime

