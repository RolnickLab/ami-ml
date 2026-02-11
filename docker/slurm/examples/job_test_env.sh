#!/bin/bash
#SBATCH --job-name=test_env_setup
#SBATCH --output=test_env_setup_%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=main

# Test job to verify environment setup similar to real SLURM jobs
echo "Testing environment setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Check if we can load conda
echo "Checking conda..."
conda --version

# Create and activate a test environment
echo "Creating test conda environment..."
conda create -n test-env python=3.9 -y
conda activate test-env

echo "Python in conda env:"
which python
python --version

# Test if Poetry is available
echo "Testing Poetry..."
poetry --version || echo "Poetry not available in PATH"

# Check if workspace is mounted
echo "Checking workspace files..."
ls -la /workspace/

echo "Environment test completed successfully!"
