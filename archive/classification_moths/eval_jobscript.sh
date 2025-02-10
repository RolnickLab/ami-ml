#!/bin/bash
#SBATCH --partition=main                      # Ask for main job
#SBATCH --cpus-per-task=6                     # Ask for 6 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Copy your dataset on the compute node
start=`date +%s`
rpath=miniscratch/transit_datasets/restricted/inat_users/inat/iNat

tar -xf /miniscratch/transit_datasets/restricted/inat_users/inat/moth_data-archived.tar -C $SLURM_TMPDIR 

end=`date +%s`
runtime=$((end-start))
echo 'Time taken for unarchiving the data in seconds' $runtime


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python eval_alone.py --data_path $SLURM_TMPDIR/$rpath --config_file config/01-config.json
