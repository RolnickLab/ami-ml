#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Test job to verify GPU availability
echo "Testing GPU availability..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi found, checking GPUs..."
    nvidia-smi
else
    echo "nvidia-smi not found - GPU passthrough may not be configured"
fi

# Test PyTorch GPU availability
cd /workspace
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda activate ami-ml 2>/dev/null || echo "ami-ml environment not found"

python3 << EOF
import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA GPUs available")
except ImportError:
    print("PyTorch not installed")
    sys.exit(0)
EOF

echo "GPU test completed!"
