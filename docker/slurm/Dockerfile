FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SLURM_VERSION=23.02.7

# Install basic dependencies and SLURM dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gcc \
    make \
    build-essential \
    libmunge-dev \
    libmunge2 \
    munge \
    libssl-dev \
    libpam0g-dev \
    libnuma-dev \
    libhwloc-dev \
    lua5.3 \
    liblua5.3-dev \
    libreadline-dev \
    libjson-c-dev \
    libibmad-dev \
    libibumad-dev \
    libpmix-dev \
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

# Create slurm user and munge user
RUN groupadd -r slurm --gid=990 && \
    useradd -r -g slurm --uid=990 slurm && \
    groupadd -r munge --gid=991 && \
    useradd -r -g munge --uid=991 munge

# Download and install SLURM
WORKDIR /tmp
RUN wget https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 && \
    tar xjf slurm-${SLURM_VERSION}.tar.bz2 && \
    cd slurm-${SLURM_VERSION} && \
    ./configure --prefix=/usr --sysconfdir=/etc/slurm --with-pmix && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf slurm-${SLURM_VERSION}*

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
COPY docker/slurm/gres.conf /etc/slurm/gres.conf

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
