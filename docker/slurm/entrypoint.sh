#!/bin/bash

set -e

# Generate munge key if it doesn't exist
if [ ! -f /etc/munge/munge.key ]; then
    echo "Generating munge key..."
    dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi

# Ensure proper permissions
chown -R munge:munge /etc/munge /var/log/munge /run/munge
chown -R slurm:slurm /var/spool/slurm /var/log/slurm

# Start munge
echo "Starting munge..."
sudo -u munge /usr/sbin/munged --force

# Wait for munge to start
sleep 2

# Check what role this container should play
ROLE=${SLURM_ROLE:-controller}

if [ "$ROLE" = "controller" ]; then
    echo "Starting SLURM controller (slurmctld)..."
    
    # Create state directory if it doesn't exist
    mkdir -p /var/spool/slurm/ctld
    chown slurm:slurm /var/spool/slurm/ctld
    
    # Start slurmctld
    exec /usr/sbin/slurmctld -D -vvv
    
elif [ "$ROLE" = "compute" ]; then
    echo "Starting SLURM compute node (slurmd)..."
    
    # Wait for controller to be ready
    sleep 5
    
    # Create spool directory if it doesn't exist
    mkdir -p /var/spool/slurm/d
    chown slurm:slurm /var/spool/slurm/d
    
    # Start slurmd
    exec /usr/sbin/slurmd -D -vvv
else
    echo "Unknown SLURM_ROLE: $ROLE"
    exit 1
fi
