#!/bin/bash

# print the commands
set -o xtrace

# Run docker image as an argument but use graspvla:latest by default
DATASET_PATH=${1:-"datasets"}
DOCKER_IMAGE=${2:-"graspvla:latest"}
USER_HOME=${3:-"/home/user"}

USER=$(whoami)

# Mount the current path to /workspace
# Find all NVIDIA devices and create device mappings
NVIDIA_DEVICES=""
# Add NVIDIA GPU devices
for device in /dev/nvidia[0-9]*; do
    if [ -e "$device" ]; then
        NVIDIA_DEVICES+="--device=$device "
    fi
done

# Add NVIDIA control device if it exists
if [ -e "/dev/nvidiactl" ]; then
    NVIDIA_DEVICES+="--device=/dev/nvidiactl "
fi

# Add NVIDIA UVM devices if they exist
for device in /dev/nvidia-uvm*; do
    if [ -e "$device" ]; then
        NVIDIA_DEVICES+="--device=$device "
    fi
done

# Add NVIDIA modeset device if it exists
if [ -e "/dev/nvidia-modeset" ]; then
    NVIDIA_DEVICES+="--device=/dev/nvidia-modeset "
fi

docker run \
    --gpus all \
    --shm-size=32g \
    -itd \
    --name=dexter \
    --network=host \
    -v "$(pwd):/workspace" \
    -v "${DATASET_PATH}:/datasets" \
    $NVIDIA_DEVICES \
    "$DOCKER_IMAGE"
