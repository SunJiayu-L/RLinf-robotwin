#!/usr/bin/env bash
# Resume the 5-task joint PPO run (initialized from the mode1-hb07-joint04
# merged checkpoint) from its global_step_50 checkpoint -- the first attempt
# hit Ray OOM at step 50 with --memory=800000. This time request 1,000,000
# MiB (1TB) to give more headroom for the 5-task multitask rollout's
# per-EnvWorker memory growth. Uses runner.resume_dir (this checkpoint has
# full training state incl. optimizer, unlike the merge weights).
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <image> [job-name]" >&2
    exit 2
fi

IMAGE="$1"
JOB_NAME="${2:-5task-ppo-mode1hb07j04-resume50-4g-m1000-$(date +%m%d-%H%M)}"
GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin"
RESUME_CKPT="${GPFS_ROOT}/rlinf-robotwin-runs/5task-ppo-mode1hb07j04-128env-4g-m800-0805-2250/5task_ppo_mode1hb07j04_128env_4gpu/checkpoints/global_step_50"
RUN_ROOT="${GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"
# Pin to a specific node via NODE_TAG env var, e.g.:
#   NODE_TAG=gpu-l-lg-cmc-h-h200-0256.host.h.pjlab.org.cn ./this_script.sh ...
# Only 4 nodes are currently usable (per muyao):
#   gpu-l-lg-cmc-h-h200-{0256,0257,0258,0260}.host.h.pjlab.org.cn
NODE_TAG="${NODE_TAG:-}"

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

SUBMIT_ARGS=(
    --name="$JOB_NAME"
    --task-type=normal
    --charged-group=eailabagent_gpu
    --private-machine=group
    --gpu=4
    --cpu=32
    --memory=1000000
    --image="$IMAGE"
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs
    -e DISTRIBUTED_JOB=true
    -e RLINF_RUN_ROOT="$RUN_ROOT"
)
if [[ -n "$NODE_TAG" ]]; then
    SUBMIT_ARGS+=(--positive-tags "node/${NODE_TAG}")
fi

rjob submit \
    "${SUBMIT_ARGS[@]}" \
    -- bash -exc '
        set -euo pipefail
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
        export VENV_PATH=/opt/venvs/openpi-robotwin-cu124
        export RESUME_CKPT=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/5task-ppo-mode1hb07j04-128env-4g-m800-0805-2250/5task_ppo_mode1hb07j04_128env_4gpu/checkpoints/global_step_50
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

        test -f "${RESUME_CKPT}/actor/dcp_checkpoint/.metadata"
        test -f "${RESUME_CKPT}/actor/model_state_dict/full_weights.pt"
        cd "${REPO_PATH}"
        ray stop --force 2>/dev/null || true
        ray start --head --num-cpus=32 --num-gpus=4
        trap "status=\$?; ray stop --force 2>/dev/null || true; exit \$status" EXIT
        mkdir -p "${RLINF_RUN_ROOT}"
        CMD=(python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py"
             --config-path "${EMBODIED_PATH}/config/"
             --config-name moe_ppo_5task_bbh_ca_cb_hb_mpa_128env_4gpu
             "runner.logger.log_path=${RLINF_RUN_ROOT}"
             "runner.logger.experiment_name=5task_ppo_mode1hb07j04_resume50_128env_4gpu"
             "runner.max_steps=500"
             "runner.resume_dir=${RESUME_CKPT}"
             "env.train.video_cfg.save_video=False"
             "env.eval.video_cfg.save_video=False"
             "env.train.task_config.eval_video_log=false"
             "env.eval.task_config.eval_video_log=false"
             "env.train.task_config.collect_data=false"
             "env.eval.task_config.collect_data=false"
             "env.train.task_config.save_freq=null"
             "env.eval.task_config.save_freq=null")
        printf "%q " "${CMD[@]}" > "${RLINF_RUN_ROOT}/run_embodiment.log"
        printf "\n" >> "${RLINF_RUN_ROOT}/run_embodiment.log"
        "${CMD[@]}" 2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
    '
