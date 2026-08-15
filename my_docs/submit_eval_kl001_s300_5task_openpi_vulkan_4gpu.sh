#!/usr/bin/env bash
# Evaluate the handover_block KL beta=0.001 checkpoint (global_step_300) on all
# 5 tasks from the joint reference policy's task set (bbh, ca, cb, hb, mpa),
# to see whether the reference-KL regularization preserved multi-task
# capability, not just handover_block's own success rate.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <image> [job-name]" >&2
    exit 2
fi

IMAGE="$1"
JOB_NAME="${2:-eval-kl001-s300-5task-4g-m400-$(date +%m%d-%H%M)}"
GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin"
CKPT="${GPFS_ROOT}/rlinf-robotwin-runs/handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155/handover_block_ppo_kl001_fromhb200_128env_4gpu/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
RUN_ROOT="${GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"

if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Job name must contain only lowercase letters, digits, and hyphens." >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="$no_proxy"
# GPFS2 may not be mounted on the development machine after control-plane
# maintenance. Keep the definitive checkpoint check inside the rjob, where
# the gpfs2 mount is attached by rjob.
mkdir -p "$RUN_ROOT"

rjob submit \
    --name="$JOB_NAME" \
    --task-type=normal \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --gpu=4 \
    --cpu=32 \
    --memory=400000 \
    --image="$IMAGE" \
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin \
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs \
    -e DISTRIBUTED_JOB=true \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -- bash -exc '
        set -euo pipefail
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
        export VENV_PATH=/opt/venvs/openpi-robotwin-cu124
        export CKPT=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155/handover_block_ppo_kl001_fromhb200_128env_4gpu/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt
        source "${VENV_PATH}/bin/activate"
        export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
        export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
        export ROBOT_PLATFORM=ALOHA
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
        export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
        export RLINF_NODE_RANK=0
        export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

        test -f "${CKPT}"
        cd "${REPO_PATH}"
        mkdir -p "${RLINF_RUN_ROOT}"

        for TASK in bbh ca cb hb mpa; do
            echo "=== Evaluating eval_kl001_s300_${TASK} ===" | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
            ray stop --force 2>/dev/null || true
            ray start --head --num-cpus=32 --num-gpus=4
            python "${REPO_PATH}/examples/embodiment/eval_embodied_agent.py" \
                --config-path "${EMBODIED_PATH}/config/" \
                --config-name "eval_kl001_s300_${TASK}" \
                "runner.logger.log_path=${RLINF_RUN_ROOT}" \
                "runner.ckpt_path=${CKPT}" \
                2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
            ray stop --force 2>/dev/null || true
        done
    '
