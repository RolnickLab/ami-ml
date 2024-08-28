#!/bin/bash
#SBATCH --job-name=gradio_demo
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu              # Ask for long-cpu job
#SBATCH --cpus-per-task=1                 # Ask for 1 CPUs
#SBATCH --mem=5G                          # Ask for 300 GB of RAM
#SBATCH --output=gradio_demo_%j.out

# 1. Load the required modules
module load miniconda/3

# 2. Load your environment
conda activate ami-ml

# 3. Run the demo
gradio global_moth_model/gradio_demo.py
