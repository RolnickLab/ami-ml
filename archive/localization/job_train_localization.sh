#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=25G                             # Ask for 25 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 3. Copy your dataset on the compute node
start=`date +%s`
cp -r /home/mila/a/aditya.jain/scratch/Localization $SLURM_TMPDIR

end=`date +%s`
runtime=$((end-start))
echo 'Time taken for copying the data in seconds' $runtime
ls $SLURM_TMPDIR


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python train_localization.py --data_path $SLURM_TMPDIR \
--model_type ssd300_vgg16 \
--pretrained DEFAULT \
--batch_size 16 \
--train_per 85 \
--num_epochs 50 \
--early_stop 4 \
--save_path /home/mila/a/aditya.jain/logs/v5_localization_model_ssd_ \
--wandb_project Localization-Model \
--wandb_entity moth-ai 



