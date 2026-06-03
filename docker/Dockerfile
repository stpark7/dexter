FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Copy uv binary from astral-sh/uv:0.6.14
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
# CUDA environment
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# CUDA architecture (H100, A6000, and A100)
ARG CUDA_ARCHITECTURES=90;86;80;
ENV TORCH_CUDA_ARCH_LIST="9.0 8.6 8.0"
# UV configuration
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PROJECT_ENVIRONMENT=/python/.venv
ENV UV_PYTHON_PREFERENCE=only-managed

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    vim \
    unzip \
    curl \
    tmux \
    libatlas-base-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgflags-dev \
    libegl1 \
    libgl1 \
    libgomp1 \
    libopenblas-dev \
    protobuf-compiler \
    sudo \
    vim-tiny \
    wget \
    openssh-server \
    libsm6 \
    libxext6 \
    openmpi-bin \
    ninja-build \
    libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# Create non root user and setup environment.
# USERNAME/UID/GID default to a generic "user"; pass the host user's ids at
# build time (e.g. --build-arg UID=$(id -u) --build-arg GID=$(id -g)) so files
# created in mounted volumes are owned by the host user.
ARG USERNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && useradd -m -d /home/${USERNAME} -g ${GID} -G sudo -u ${UID} ${USERNAME}
RUN usermod -aG sudo ${USERNAME}
# Set user password
RUN echo "${USERNAME}:${USERNAME}" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by amending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install python
RUN uv python install 3.11

# Install dependencies
WORKDIR /python
COPY pyproject.toml README.md .python-version /python/
COPY src /python/src
COPY thirdparty /python/thirdparty
ENV PATH="/python/.venv/bin:$PATH"

# Install Python dependencies with CUDA 12.8 PyTorch 2.8.0.
# uv sync creates the venv and does a two-phase install so torch is in place
# before the source-built CUDA extensions (pointops, chamferdist, pytorch3d, ...).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Change ownership of Python environment to the created user
RUN chown -R ${UID}:${GID} /python

WORKDIR /workspace

CMD /bin/bash
