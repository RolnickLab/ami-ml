# SLURM Docker Environment for Local Testing

This directory contains a Docker Compose setup that simulates a SLURM environment for local testing of job scripts before submitting to DRAC/Compute Canada HPC clusters.

## Overview

The environment consists of:
- **SLURM Controller** (`slurmctld`): Manages job scheduling and cluster state
- **Compute Node** (`c1`): Single compute node with 8 CPUs, 32GB RAM, and optional GPU support
- **Partitions**: Configured to match DRAC's partition structure:
  - `main`: Default partition for general compute jobs (max 48h)
  - `long`: For longer-running jobs (unlimited time)
  - `long-cpu`: CPU-only jobs with unlimited time

## Configuration

The SLURM configuration is based on [DRAC/Compute Canada documentation](https://docs.alliancecan.ca/wiki/Running_jobs) and matches the behavior of their clusters as closely as possible:

- **Authentication**: MUNGE for inter-process communication
- **Scheduling**: Backfill scheduler with fair-share policies
- **Resource Management**: Cgroup-based process tracking and resource constraints
- **GPU Support**: Optional GPU passthrough using NVIDIA Docker runtime

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **NVIDIA Docker** (optional, for GPU support):
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Quick Start

### 1. Build and Start the Environment

```bash
# From the repository root
docker-compose -f docker-compose.slurm.yml build
docker-compose -f docker-compose.slurm.yml up -d
```

Wait for both containers to start (approximately 30 seconds):

```bash
# Check container status
docker-compose -f docker-compose.slurm.yml ps

# View logs
docker-compose -f docker-compose.slurm.yml logs -f
```

### 2. Access the SLURM Controller

```bash
# Enter the controller container
docker exec -it ami-ml-slurmctld bash
```

### 3. Check SLURM Status

```bash
# Check cluster info
sinfo

# Check node status
scontrol show nodes

# Check partition configuration
scontrol show partitions
```

Expected output from `sinfo`:
```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up 2-00:00:00      1   idle c1
long         up   infinite      1   idle c1
long-cpu     up   infinite      1   idle c1
```

### 4. Submit a Test Job

```bash
# Inside the controller container, navigate to workspace
cd /workspace

# Submit a simple test job
sbatch docker/slurm/examples/job_hello.sh

# Check job status
squeue

# View job output (after completion)
cat hello_slurm_*.out
```

### 5. Test with Real Job Scripts

The repository's job scripts can be tested by copying and modifying them:

```bash
# Example: Test the training classifier job
# Note: You may need to adjust paths and reduce epochs for quick testing

# Copy an existing job script
cp research/order_level_classifier/job_train_classifier.sh docker/slurm/examples/job_test_quick.sh

# Edit to reduce runtime (e.g., fewer epochs, smaller dataset)
# Then submit
sbatch docker/slurm/examples/job_test_quick.sh
```

## Testing Workflow

### Environment Setup Test

Test that the conda environment and Poetry work correctly:

```bash
sbatch docker/slurm/examples/job_test_env.sh
squeue  # Check job status
cat test_env_setup_*.out  # View output after completion
```

### AMI-ML Package Test

Test that the ami-ml commands are available:

```bash
sbatch docker/slurm/examples/job_test_ami_ml.sh
```

## Common SLURM Commands

### Job Management

```bash
# Submit a job
sbatch job_script.sh

# Check job queue
squeue
squeue -u $USER  # Your jobs only

# Get detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# View completed job info
sacct -j <job_id>
```

### Node and Partition Info

```bash
# List partitions
sinfo -s

# Show detailed partition info
scontrol show partition main

# Show node details
scontrol show node c1
```

### Resource Monitoring

```bash
# Monitor job output in real-time
tail -f <output_file>.out

# Check cluster load
sinfo -N -l
```

## Testing GPU Jobs

If you have NVIDIA GPUs and the NVIDIA Docker runtime installed:

```bash
# Submit a GPU job
sbatch --gres=gpu:1 docker/slurm/examples/job_gpu_test.sh

# Check GPU availability
scontrol show node c1 | grep Gres
```

To test without GPU (CPU-only smoke tests):
- Remove or comment out the `#SBATCH --gres=gpu:*` line
- Reduce training to 1-2 epochs
- Use smaller batch sizes

## Adapting Real Job Scripts

When adapting real job scripts from `research/` or `scripts/`:

1. **Module Loading**: The container uses conda directly, not `module load`
   ```bash
   # Real SLURM:
   module load miniconda/3
   conda activate ami-ml
   
   # Docker SLURM:
   eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
   conda activate ami-ml
   ```

2. **Data Paths**: Use paths relative to `/workspace` (project root)
   ```bash
   # Real SLURM (uses environment variables):
   --train_webdataset $ORDER_WBDS_ALL
   
   # Docker SLURM (use local paths or set env vars):
   --train_webdataset /workspace/data/train_*.tar
   ```

3. **Temporary Storage**: `$SLURM_TMPDIR` works the same way in the container

4. **Resource Requests**: Adjust to fit the container limits (8 CPUs, 32GB RAM, 2 GPUs)

## Troubleshooting

### SLURM Controller Not Starting

```bash
# Check controller logs
docker-compose -f docker-compose.slurm.yml logs slurm-controller

# Common issues:
# - Munge key not generated properly
# - Configuration file errors

# Restart the environment
docker-compose -f docker-compose.slurm.yml down -v
docker-compose -f docker-compose.slurm.yml up -d
```

### Jobs Stuck in Pending State

```bash
# Check why job is pending
scontrol show job <job_id>

# Common reasons:
# - Requested resources exceed node capacity
# - Wrong partition name
# - Compute node not responding

# Check node state
scontrol show node c1

# If node is down, set it to idle
scontrol update nodename=c1 state=idle
```

### Permission Issues

```bash
# Ensure workspace is accessible
ls -la /workspace

# Check SLURM user permissions
id slurm
```

### Conda Environment Issues

```bash
# If conda environment doesn't exist, create it:
conda create -n ami-ml python=3.9 -y
conda activate ami-ml
cd /workspace
poetry install
```

## Stopping the Environment

```bash
# Stop containers
docker-compose -f docker-compose.slurm.yml down

# Stop and remove volumes (clean state)
docker-compose -f docker-compose.slurm.yml down -v
```

## Advanced Configuration

### Customizing Node Resources

Edit `docker/slurm/slurm.conf` to adjust resources:

```conf
NodeName=c1 CPUs=16 RealMemory=64000 Gres=gpu:4 State=UNKNOWN
```

Then rebuild:
```bash
docker-compose -f docker-compose.slurm.yml build
docker-compose -f docker-compose.slurm.yml up -d
```

### Adding More Compute Nodes

Edit `docker-compose.slurm.yml` to add more nodes, and update `slurm.conf` accordingly.

### Persistent Storage

Job outputs and SLURM logs are stored in Docker volumes. To use host directories instead:

```yaml
volumes:
  - ./slurm-logs:/var/log/slurm
  - ./slurm-spool:/var/spool/slurm
```

## Differences from Real DRAC SLURM

1. **Scale**: Single node vs. multi-node cluster
2. **Modules**: No `module load` environment modules (uses conda directly)
3. **Storage**: No scratch/project filesystem hierarchy
4. **Accounting**: No database accounting (uses flat file)
5. **Network**: No high-speed interconnect (InfiniBand)

Despite these differences, the job script syntax, resource allocation, and scheduling behavior closely match DRAC's environment.

## CI/CD Integration

This SLURM environment can be used in GitHub Actions for automated testing. See `.github/workflows/test-slurm-jobs.yml` for an example workflow.

## Resources

- [DRAC Running Jobs Documentation](https://docs.alliancecan.ca/wiki/Running_jobs)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [SLURM Docker Cluster](https://github.com/giovtorres/slurm-docker-cluster)

## Support

For issues with the Docker SLURM environment, please open an issue in the repository.
