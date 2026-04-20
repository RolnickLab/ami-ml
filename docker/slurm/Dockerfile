FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies and SLURM from Ubuntu packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gcc \
    make \
    build-essential \
    munge \
    slurm-wlm \
    slurm-wlm-basic-plugins \
    slurmd \
    slurmctld \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    sudo \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create required directories
RUN mkdir -p /var/spool/slurm/ctld \
    /var/spool/slurm/d \
    /var/log/slurm \
    /etc/slurm \
    /run/munge

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

# Install Poetry
RUN pip3 install poetry

# Copy SLURM configuration files
COPY docker/slurm/slurm.conf /etc/slurm/slurm.conf
COPY docker/slurm/cgroup.conf /etc/slurm/cgroup.conf
# COPY docker/slurm/gres.conf /etc/slurm/gres.conf  # Uncomment for GPU support

# Set proper permissions
RUN chown -R slurm:slurm /var/spool/slurm /var/log/slurm && \
    chown -R munge:munge /etc/munge /var/log/munge /run/munge && \
    chmod 700 /etc/munge /var/log/munge /run/munge

# Copy entrypoint script
COPY docker/slurm/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["slurmctld"]
