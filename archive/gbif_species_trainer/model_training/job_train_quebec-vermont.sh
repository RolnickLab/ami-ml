#!/bin/bash
#SBATCH --partition=main      # Ask for main job
#SBATCH --cpus-per-task=6     # Ask for 6 CPUs
#SBATCH --gres=gpu:rtx8000:2  # Ask for 2 GPU
#SBATCH --mem=15G             # Ask for 15 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Copy your dataset on the compute node
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_quebec-vermont/train/train-500*.tar $SLURM_TMPDIR
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_quebec-vermont/val/val-500*.tar $SLURM_TMPDIR
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_quebec-vermont/test/test-500*.tar $SLURM_TMPDIR


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python 04-train_model_v2_resolution.py \
--train_webdataset_url "$SLURM_TMPDIR/train-500-{000000..000465}.tar" \
--val_webdataset_url "$SLURM_TMPDIR/val-500-{000000..000061}.tar" \
--test_webdataset_url "$SLURM_TMPDIR/test-500-{000000..000093}.tar" \
--config_file config/01-config_quebec-vermont.json \
--dataloader_num_workers 6

