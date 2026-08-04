#!/usr/bin/env bash

set -euo pipefail

IMAGE_TAG="${1:-registry.h.pjlab.org.cn/ailab-eailabagent-eailabagent_gpu/jiayusun-rlinf-robotwin:openpi-robotwin-vulkan-20260728}"

docker build \
  --network host \
  --build-arg http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
  --build-arg https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
  --build-arg no_proxy="${no_proxy:-${NO_PROXY:-}}" \
  --build-arg NO_PROXY="${NO_PROXY:-${no_proxy:-}}" \
  -f docker/Dockerfile.robotwin_vulkan_openpi \
  -t "${IMAGE_TAG}" \
  .

docker push "${IMAGE_TAG}"

echo "${IMAGE_TAG}"
