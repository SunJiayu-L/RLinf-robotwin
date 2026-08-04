#!/usr/bin/env bash
# Diagnostic: run the exact same resume training via rlaunch instead of rjob,
# to see whether the startup hang/raylet-kill is specific to the rjob
# container/scheduling path. rlaunch is NOT recommended for production long
# runs (worker dies if this workspace restarts) — this is for diagnosis /
# short verification only. See my_docs/2026-07-28-ppo50to500-resume-startup-hang-investigation.md

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <image>" >&2
    exit 2
fi

IMAGE="$1"
JOB_NAME="rlaunch-ppo50to500-resume-$(date +%m%d-%H%M)"
RUN_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/${JOB_NAME}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="$no_proxy"
mkdir -p "$RUN_ROOT"

rlaunch \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --gpu=4 \
    --cpu=32 \
    --memory=800000 \
    --image="$IMAGE" \
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin \
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs \
    -e DISTRIBUTED_JOB=true \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -d -- bash -exc '
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
        export BOOTSTRAP_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/bootstrap
        export VENV_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/venvs/openpi-robotwin-cu124
        test -f "${VENV_PATH}/bin/activate"
        source "${VENV_PATH}/bin/activate"
        export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
        export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
        export OPENPI_DATA_HOME="${BOOTSTRAP_PATH}/root/.cache/openpi"
        export RESUME_CKPT=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin/logs/20260725-11:32:49-moe_sft20k_handover_block_ppo150_128env_4gpu/moe_sft20k_handover_block_ppo150_128env_4gpu/checkpoints/global_step_50
        export ROBOT_PLATFORM=ALOHA
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
        export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
        export TORCH_DISTRIBUTED_DEBUG=OFF
        export RLINF_NODE_RANK=0
        export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

        cd "${REPO_PATH}"
        ray stop --force 2>/dev/null || true
        ray start --head --num-cpus=32 --num-gpus=4
        trap "status=\$?; ray stop --force 2>/dev/null || true; exit \$status" EXIT
        test -f "${RESUME_CKPT}/actor/dcp_checkpoint/.metadata"
        test -f "${RESUME_CKPT}/actor/model_state_dict/full_weights.pt"
        test -f "${RESUME_CKPT}/robotwin_moe_demo_clean/norm_stats.json"
        mkdir -p "${RLINF_RUN_ROOT}"
        CMD=(python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py"
             --config-path "${EMBODIED_PATH}/config/"
             --config-name moe_sft20k_handover_block_ppo150_128env_4gpu
             "runner.logger.log_path=${RLINF_RUN_ROOT}"
             "runner.logger.experiment_name=moe_sft20k_handover_block_ppo50to500_128env_4gpu_resume_rlaunch"
             "runner.max_steps=500"
             "runner.resume_dir=${RESUME_CKPT}")
        printf "%q " "${CMD[@]}" > "${RLINF_RUN_ROOT}/run_embodiment.log"
        printf "\n" >> "${RLINF_RUN_ROOT}/run_embodiment.log"
        "${CMD[@]}" 2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
    '

echo "RUN_ROOT=${RUN_ROOT}"
