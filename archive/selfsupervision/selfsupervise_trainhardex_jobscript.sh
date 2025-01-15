#!/bin/bash
#SBATCH --partition=unkillable              # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=30G                             # Ask for 5 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Copy your dataset on the compute node
start=`date +%s`
cp -r /network/datasets/imagenet.var/imagenet_torchvision/val/ $SLURM_TMPDIR 
cp -r /home/mila/a/aditya.jain/scratch/selfsupervise_data/hard_examples/ $SLURM_TMPDIR
end=`date +%s`
runtime=$((end-start))
echo 'Time taken for copying the data in seconds' $runtime 


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python -u train_hardex.py --data_path $SLURM_TMPDIR
