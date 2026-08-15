#!/usr/bin/env bash
# Generic 5-task (bbh/ca/cb/hb/mpa) fixed-seed eval submitter. Reuses the
# existing eval_kl001_s300_{bbh,ca,cb,hb,mpa}.yaml configs as templates --
# their baked-in ckpt_path is irrelevant since it's always overridden via
# runner.ckpt_path on the CLI, so the same 5 yaml files work for ANY
# checkpoint (weight-merge candidates, baselines, etc.), not just the
# kl001-s300 run they were originally named after.
#
# Usage: submit_eval_5task_generic.sh <image> <ckpt_path> <job-name>
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <image> <ckpt_path> <job-name>" >&2
    exit 2
fi

IMAGE="$1"
CKPT_ARG="$2"
JOB_NAME="$3"
GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin"
RUN_ROOT="${GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"

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
    --gpu=4 \
    --cpu=32 \
    --memory=400000 \
    --image="$IMAGE" \
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin \
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs \
    -e DISTRIBUTED_JOB=true \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -e EVAL_CKPT_PATH="$CKPT_ARG" \
    -- bash -exc '
        set -euo pipefail
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
        export VENV_PATH=/opt/venvs/openpi-robotwin-cu124
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

        test -f "${EVAL_CKPT_PATH}"
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
                "runner.ckpt_path=${EVAL_CKPT_PATH}" \
                2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
            ray stop --force 2>/dev/null || true
        done
    '
