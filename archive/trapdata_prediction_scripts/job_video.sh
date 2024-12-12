#!/bin/bash
#SBATCH --partition=main                # Ask for main job
#SBATCH --cpus-per-task=2               # Ask for 2 CPUs
#SBATCH --gres=gpu:1                    # Ask for 1 GPU
#SBATCH --mem=2G                        # Ask for 2 GB of RAM


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Run the script
python make_video.py \
--data_dir '/home/mila/a/aditya.jain/scratch/TrapData_QuebecVermont_2022/Quebec/' \
--image_folder '2022_05_18' \
--frame_rate 5 \
--scale_factor 0.4 \
--region 'Quebec'




