#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

echo "Running as rootful docker!"
unset DOCKER_HOST

docker context ls

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

export USER_NAME=$(whoami)
DEVICE=$1
MASTER_HOST_NAME=$2
GLOBAL_RANK=$3

echo ""
echo "Using GPU devices: ${DEVICE}"
echo "User: ${USER_NAME}"
echo ""
echo "Master host: ${MASTER_HOST_NAME}"
echo "Global rank: ${GLOBAL_RANK}"
echo ""



#--------------------------------------------------------------------
#
#   Distributed Args
#
#--------------------------------------------------------------------

LOCAL_HOST_NAME=$(hostname)

# These are necessary to make the connection
DEVICE_NAME=`echo ${DEVICE} | tr "," "-"`

IMAGE_NAME=stunngan-${LOCAL_HOST_NAME}-${DEVICE_NAME}
MASTER_ADDR=stunngan-${MASTER_HOST_NAME}
MASTER_PORT=24000

# Configured by the Swarm !
NETWORK_NAME=net-overlay
DDP_INTERFACE=eth0

WORLD_SIZE=2
LOCAL_RANK=0
RANK=$3


docker run \
    -it --rm \
    --user $(id -u):$(id -g) \
    --name "${IMAGE_NAME}" \
    --hostname "${IMAGE_NAME}" \
    --network ${NETWORK_NAME} \
    --gpus all \
    --privileged \
    --shm-size 8g \
    --device /dev/fuse \
    -v "${HOME}/.netrc":/root/.netrc \
    -v "${CWD}/..":/workspace/${PROJECT_NAME} \
    -v "/mnt/scratch/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.scratch \
    -e CUDA_VISIBLE_DEVICES="${DEVICE}" \
    -e MASTER_ADDR=${MASTER_ADDR} \
    -e MASTER_PORT=${MASTER_PORT} \
    -e WORLD_SIZE=${WORLD_SIZE} \
    -e NODE_RANK=${RANK} \
    -e NCCL_SOCKET_IFNAME=${DDP_INTERFACE} \
    -e NCCL_ASYNC_ERROR_HANDLING=1 \
    ${IMAGE_TAG} 