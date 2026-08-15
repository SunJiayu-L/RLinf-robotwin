#!/usr/bin/env bash
# Tiny diagnostic job: pin to a specific node, print nvidia-smi + free -h +
# df -h, then exit. Used to check real GPU/memory status of a node before
# scheduling a heavy job there.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <image> <node-fqdn> [job-name]" >&2
    exit 2
fi

IMAGE="$1"
NODE_TAG="$2"
JOB_NAME="${3:-node-diag-$(echo "$NODE_TAG" | grep -oE '[0-9]{4}')-$(date +%m%d-%H%M)}"
RUN_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/${JOB_NAME}"

if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Job name must contain only lowercase letters, digits, and hyphens." >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="$no_proxy"
mkdir -p "$RUN_ROOT"

rjob submit \
    --name="$JOB_NAME" \
    --task-type=normal \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --gpu=1 \
    --cpu=2 \
    --memory=8000 \
    --image="$IMAGE" \
    --positive-tags "node/${NODE_TAG}" \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -- bash -c '
        echo "=== node: '"${NODE_TAG}"' ==="
        echo "--- nvidia-smi ---"
        nvidia-smi || echo "nvidia-smi failed"
        echo "--- free -h ---"
        free -h
        echo "--- df -h / ---"
        df -h / 2>/dev/null || true
    '
