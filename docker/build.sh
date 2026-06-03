#!/bin/bash

# Default image name and tag
IMAGE_NAME="dexter"
TAG="latest"

# Default user settings: generic username with the host user's uid/gid so that
# files created in mounted volumes stay owned by the host user.
USERNAME="user"
UID_ARG=$(id -u)
GID_ARG=$(id -g)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --username)
      USERNAME="$2"
      shift 2
      ;;
    --uid)
      UID_ARG="$2"
      shift 2
      ;;
    --gid)
      GID_ARG="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--image-name IMAGE_NAME] [--tag TAG] [--username USERNAME] [--uid UID] [--gid GID]"
      echo "  --image-name   Docker image name (default: dexter)"
      echo "  --tag          Docker image tag (default: latest)"
      echo "  --username     Container user name (default: user)"
      echo "  --uid          Container user uid (default: host \$(id -u))"
      echo "  --gid          Container user gid (default: host \$(id -g))"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the Docker image
echo "Building Docker image: ${IMAGE_NAME}:${TAG} (user=${USERNAME} uid=${UID_ARG} gid=${GID_ARG})"
docker build \
    --build-arg USERNAME="${USERNAME}" \
    --build-arg UID="${UID_ARG}" \
    --build-arg GID="${GID_ARG}" \
    -t "${IMAGE_NAME}:${TAG}" -f docker/Dockerfile .
