#!/bin/bash
#SBATCH --job-name=eval_global_model_gbif_test
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --partition=main                # Ask for main job
#SBATCH --cpus-per-task=8               # Ask for 8 CPUs
#SBATCH --gres=gpu:rtx8000:2            # Ask for 2 GPU
#SBATCH --output=eval_global_model_gbif_test_%j.out

# 1. Load the required modules
module load miniconda/3

# 2. Load your environment
conda activate ami-ml

# Keep track of time
SECONDS=0

# 3. Copy your dataset on the compute node
cp /home/mila/a/aditya.jain/scratch/global_model/webdataset/test/*.tar $SLURM_TMPDIR

# Print time taken to execute the script
echo "Time taken to copy the dataset: $((SECONDS/60)) minutes"

# 4. Launch your job
python global_moth_model/gbif_test_eval/gbif_test_eval.py

# Print time taken to execute the script
echo "Time taken to run test evaluation: $((SECONDS/60)) minutes"



