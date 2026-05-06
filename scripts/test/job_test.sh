#!/bin/bash
#SBATCH --job-name=ami_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:05:00
#SBATCH --output=ami_test_%j.out
#SBATCH --account=def-drolnick

BASE_DIR="/home/melabbas/projects/def-drolnick/melabbas/ami-ml"

cd "$BASE_DIR"
source .venv/bin/activate

echo "=== ami-ml test job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo "ami-dataset: $(ami-dataset --version 2>&1 || echo 'not found')"
echo "=== done ==="
