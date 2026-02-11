#!/bin/bash
#SBATCH --job-name=test_ami_ml_training
#SBATCH --output=test_ami_ml_training_%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=main

# Minimal training test to verify ami-ml package works
echo "Testing ami-ml training setup..."
echo "Job ID: $SLURM_JOB_ID"

# Set working directory
cd /workspace

# Load conda and activate environment
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda activate ami-ml || {
    echo "Creating ami-ml environment..."
    conda create -n ami-ml python=3.9 -y
    conda activate ami-ml
    
    # Install dependencies with Poetry
    poetry install
}

# Load environment variables
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# Test that ami-ml commands are available
echo "Testing ami-ml commands..."
ami-classification --help || echo "ami-classification command not available"
ami-dataset --help || echo "ami-dataset command not available"

echo "Test completed successfully!"
