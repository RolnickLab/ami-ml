#!/bin/bash
#SBATCH --job-name=hello_slurm
#SBATCH --output=hello_slurm_%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

# Simple test job to verify SLURM is working
echo "Hello from SLURM!"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Python version:"
python3 --version
echo "Conda version:"
conda --version || echo "Conda not in PATH"
