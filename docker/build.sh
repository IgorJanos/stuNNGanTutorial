#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t "${IMAGE_TAG}" . || exit $?

#    --ssh default \

# install package in editable mode outside the image build to spawn egginfo with
# package metainformation. This has to be done because the original egginfo created
# during image build is overwritten by package source files mount in run.sh

# ./docker/run.sh pip install -e . --no-deps
