#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=$1
echo "Using GPU devices: ${DEVICE}"

export USER_NAME=$(whoami)
echo "User: ${USER_NAME}"


docker run \
    -it --rm \
    --name "stunngan-${DEVICE}" \
    --gpus all \
    --privileged \
    --shm-size 8g \
    --device /dev/fuse \
    -v "${HOME}/.netrc":/root/.netrc \
    -v "${CWD}/..":/workspace/${PROJECT_NAME} \
    -v "/mnt/scratch/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.scratch \
    -e CUDA_VISIBLE_DEVICES="${DEVICE}" \
    ${IMAGE_TAG} 