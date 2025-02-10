#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 1 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=25G                             # Ask for 25 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Copy your dataset on the compute node
# start=`date +%s`
rpath=home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk 

tar -xf /home/mila/a/aditya.jain/scratch/GBIF_Data/uk-moth-data_archived.tar -C $SLURM_TMPDIR 

end=`date +%s`
runtime=$((end-start))
echo 'Time taken for unarchiving the data in seconds' $runtime


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python train_uk.py --data_path $SLURM_TMPDIR/$rpath --config_file config/01-config_uk.json
